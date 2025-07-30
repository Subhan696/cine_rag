import { DataAPIClient } from "@datastax/astra-db-ts"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"
import { GoogleGenerativeAI } from "@google/generative-ai"
import fetch from "node-fetch"
import "dotenv/config"

// === ENV ===
const {
  ASTRA_DB_NAMESPACE,
  ASTRA_DB_COLLECTION,
  ASTRA_DB_API_ENDPOINT,
  ASTRA_DB_APPLICATION_TOKEN,
  GOOGLE_API_KEY,
  TMDB_API_KEY
} = process.env

// === Astra DB Setup ===
const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN!)
const db = client.db(ASTRA_DB_API_ENDPOINT!, { keyspace: ASTRA_DB_NAMESPACE! })

// === Gemini Setup ===
const genAI = new GoogleGenerativeAI(GOOGLE_API_KEY!)
const embeddingModel = genAI.getGenerativeModel({ model: "models/embedding-001" })

// === Timeout Wrapper for Gemini Embedding ===
const timeout = (ms: number) => new Promise((_, reject) => setTimeout(() => reject(new Error("Timeout")), ms))

async function getGeminiEmbedding(text: string): Promise<number[]> {
  const result = await embeddingModel.embedContent({
    content: { role: "user", parts: [{ text }] },
  })
  return result.embedding.values
}

async function getEmbeddingWithTimeout(text: string): Promise<number[]> {
  return await Promise.race([
    getGeminiEmbedding(text),
    timeout(10000) as Promise<never> // 10 seconds, ensure type never so it only rejects
  ])
}

// === Chunk Splitter ===
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 512,
  chunkOverlap: 100,
})

// === Utility ===
const sleep = (ms: number) => new Promise((res) => setTimeout(res, ms))

// === Create Collection ===
const createCollection = async () => {
  try {
    await db.createCollection(ASTRA_DB_COLLECTION!, {
      vector: { dimension: 768, metric: "cosine" },
    })
    console.log("✅ Collection created")
  } catch (err: any) {
    if (err.message.includes("already exists")) {
      console.log("ℹ️ Collection already exists")
    } else {
      throw err
    }
  }
}

// === Fetch Movies ===
async function fetchMoviesByYear(year: number, page: number = 1) {
  const url = `https://api.themoviedb.org/3/discover/movie?api_key=${TMDB_API_KEY}&language=en-US&sort_by=popularity.desc&include_adult=false&include_video=false&primary_release_year=${year}&page=${page}`
  const res = await fetch(url)
  if (!res.ok) {
    console.error("TMDB fetch failed:", await res.text())
    return []
  }
  const data = await res.json()
  return data.results || []
}

// === Fetch Watch Providers ===
async function fetchWatchProviders(movieId: number): Promise<string[]> {
  const url = `https://api.themoviedb.org/3/movie/${movieId}/watch/providers?api_key=${TMDB_API_KEY}`
  try {
    const res = await fetch(url)
    const data = await res.json()
    const usProviders = data.results?.US?.flatrate || []
    return usProviders.map((p: any) => p.provider_name)
  } catch (err) {
    console.error(`❌ Watch provider fetch failed for ${movieId}:`, err)
    return []
  }
}

// === Ingest Movies ===
const ingestMovies = async (startYear = 2000, endYear = 2025) => {
  const collection = await db.collection(ASTRA_DB_COLLECTION!)
  let counter = 0

  for (let year = startYear; year <= endYear; year++) {
    console.log(`📅 Year: ${year}`)
    for (let page = 1; page <= 500; page++) {
      console.log(`  🌐 Page: ${page}`)
      const movies = await fetchMoviesByYear(year, page)
      if (movies.length === 0) break

      for (const movie of movies) {
        const title = movie.title?.trim()
        const releaseDate = movie.release_date?.trim()
        const overview = movie.overview?.trim()
        const rating = movie.vote_average

        if (!title || !releaseDate || !overview || rating < 7.0) continue

        const providers = await fetchWatchProviders(movie.id)
        const providerText = providers.length > 0
          ? `Available on: ${providers.join(", ")}`
          : "Availability: Unknown"

        const content = `
Title: ${title}
Overview: ${overview}
Rating: ${rating}
Release Date: ${releaseDate}
${providerText}`

        const chunks = await splitter.splitText(content)

        for (const chunk of chunks) {
          try {
            const vector = await getEmbeddingWithTimeout(chunk)
            await collection.updateOne(
              {
                title,
                release_date: releaseDate,
                text: chunk
              },
              {
                $set: {
                  $vector: vector,
                  rating,
                  where_to_watch: providers,
                  source: `https://www.themoviedb.org/movie/${movie.id}`
                }
              },
              { upsert: true }
            )
            counter++
            if (counter % 100 === 0) {
              console.log(`🔄 Progress: ${counter} entries inserted`)
            }
          } catch (err) {
            console.error(`❌ Error inserting chunk for ${title}:`, err)
          }
        }
      }

      await sleep(250)
    }
  }
}

// === Main ===
const main = async () => {
  await createCollection()

  await Promise.all([
    ingestMovies(2000, 2004),
    ingestMovies(2005, 2009),
    ingestMovies(2010, 2014),
    ingestMovies(2015, 2019),
    ingestMovies(2020, 2025),
  ])
}

// === Crash Protection ===
process.on('unhandledRejection', err => {
  console.error('🔴 Unhandled Rejection:', err)
})
process.on('uncaughtException', err => {
  console.error('🔴 Uncaught Exception:', err)
})

main()
