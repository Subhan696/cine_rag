import { DataAPIClient } from "@datastax/astra-db-ts"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"
import { GoogleGenerativeAI } from "@google/generative-ai"
import fetch from "node-fetch"
import "dotenv/config"

const {
  ASTRA_DB_NAMESPACE,
  ASTRA_DB_COLLECTION,
  ASTRA_DB_API_ENDPOINT,
  ASTRA_DB_APPLICATION_TOKEN,
  GOOGLE_API_KEY,
  TMDB_API_KEY
} = process.env

const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN!)
const db = client.db(ASTRA_DB_API_ENDPOINT!, { keyspace: ASTRA_DB_NAMESPACE! })

const genAI = new GoogleGenerativeAI(GOOGLE_API_KEY!)
const embeddingModel = genAI.getGenerativeModel({ model: "models/embedding-001" })

async function getGeminiEmbedding(text: string): Promise<number[]> {
  const result = await embeddingModel.embedContent({
    content: { role: "user", parts: [{ text }] },
  });
  return result.embedding.values;
}

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 512,
  chunkOverlap: 100,
})

const sleep = (ms: number) => new Promise((res) => setTimeout(res, ms))

const createCollection = async () => {
  try {
    await db.createCollection(ASTRA_DB_COLLECTION!, {
      vector: {
        dimension: 768,
        metric: "cosine",
      },
    })
    console.log("✅ Collection created")
  } catch (e) {
    if (`${e}`.includes("already exists")) {
      console.log("ℹ️ Collection already exists")
    } else throw e
  }
}

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

async function fetchWatchProviders(movieId: number): Promise<string[]> {
  const url = `https://api.themoviedb.org/3/movie/${movieId}/watch/providers?api_key=${TMDB_API_KEY}`
  try {
    const res = await fetch(url)
    const data = await res.json()
    const usProviders = data.results?.US?.flatrate || []
    return usProviders.map((p: any) => p.provider_name)
  } catch (err) {
    console.error(`❌ Provider fetch failed for ${movieId}:`, err)
    return []
  }
}

const ingestMovies = async (startYear = 2000, endYear = 2025) => {
  const collection = await db.collection(ASTRA_DB_COLLECTION!)
  const seen = new Set<string>()

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

        if (!title || !releaseDate || !overview || rating < 6.0) continue

        const key = `${title.toLowerCase()}_${releaseDate}`
        if (seen.has(key)) continue
        seen.add(key)

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
            const vector = await getGeminiEmbedding(chunk)
            await collection.insertOne({
              $vector: vector,
              text: chunk,
              title,
              release_date: releaseDate,
              rating,
              where_to_watch: providers,
              source: `https://www.themoviedb.org/movie/${movie.id}`
            })
            console.log(`✅ Inserted: ${title}`)
          } catch (err) {
            console.error(`❌ Insert failed for ${title}:`, err)
          }
        }
      }

      await sleep(250) // Prevent hitting TMDB rate limits
    }
  }
}

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

main()
