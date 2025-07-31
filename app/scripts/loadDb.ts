import { DataAPIClient } from "@datastax/astra-db-ts"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"
import { GoogleGenerativeAI } from "@google/generative-ai"
import fetch from "node-fetch"
import dns from "dns"
import pLimit from "p-limit"
import "dotenv/config"

const {
  ASTRA_DB_NAMESPACE,
  ASTRA_DB_COLLECTION,
  ASTRA_DB_API_ENDPOINT,
  ASTRA_DB_APPLICATION_TOKEN,
  GOOGLE_API_KEY,
  TMDB_API_KEY
} = process.env

const genAI = new GoogleGenerativeAI(GOOGLE_API_KEY!)
const embeddingModel = genAI.getGenerativeModel({ model: "models/embedding-001" })

const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN!)
const db = client.db(ASTRA_DB_API_ENDPOINT!, { keyspace: ASTRA_DB_NAMESPACE! })

const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 512, chunkOverlap: 100 })

const sleep = (ms: number) => new Promise((res) => setTimeout(res, ms))

async function isOnline(): Promise<boolean> {
  return new Promise((resolve) => {
    dns.lookup("google.com", (err) => resolve(!err))
  })
}

async function waitForInternet(maxWait = 60000) {
  const start = Date.now()
  while (!(await isOnline())) {
    if (Date.now() - start > maxWait) {
      throw new Error("🛑 No internet connection for too long. Aborting.")
    }
    console.log("⏳ Waiting for internet...")
    await sleep(5000)
  }
}

async function getGeminiEmbedding(text: string): Promise<number[]> {
  const result = await embeddingModel.embedContent({
    content: { role: "user", parts: [{ text }] },
  })
  return result.embedding.values
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
  } catch {
    return []
  }
}

const createCollection = async () => {
  const res = await db.createCollection(ASTRA_DB_COLLECTION!, {
    vector: { dimension: 768, metric: "cosine" },
  })
  console.log("✅ Collection created:", res)
}

const insertWithRetry = async (collection, doc, maxRetries = 3) => {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await collection.insertOne(doc)
    } catch (err: any) {
      if (err.code?.includes("TIMEOUT") || err.code?.includes("UND_ERR_CONNECT_TIMEOUT")) {
        console.warn(`⚠️ Timeout on attempt ${i + 1}. Retrying...`)
        await waitForInternet()
        await sleep(3000 * (i + 1)) // exponential backoff
      } else {
        throw err
      }
    }
  }
  throw new Error(`❌ Max retries reached for ${doc.title}`)
}

const ingestMovies = async (startYear = 2000, endYear = 2025) => {
  const collection = await db.collection(ASTRA_DB_COLLECTION!)
  const limit = pLimit(3)

  for (let year = startYear; year <= endYear; year++) {
    console.log(`📅 Processing Year: ${year}`)

    for (let page = 1; page <= 500; page++) {
      console.log(`  🌐 Page ${page}`)
      const movies = await fetchMoviesByYear(year, page)
      if (movies.length === 0) break

      const tasks = movies.map((movie) =>
        limit(async () => {
          const title = movie.title || ""
          const overview = movie.overview || ""
          const release_date = movie.release_date || ""

          if (!title || !overview || movie.vote_average * 10 < 70) return

          const exists = await collection.findOne({ title, release_date })
          if (exists) {
            console.log(`⚠️ Skipped duplicate: "${title}" (${release_date})`)
            return
          }

          const providers = await fetchWatchProviders(movie.id)
          const providerText = providers.length > 0 ? `Available on: ${providers.join(", ")}` : "Availability: Unknown"

          const fullText = `
Title: ${title}
Overview: ${overview}
Rating: ${movie.vote_average}
Release Date: ${release_date}
${providerText}`

          const chunks = await splitter.splitText(fullText)

          for (const chunk of chunks) {
            try {
              const vector = await getGeminiEmbedding(chunk)
              await insertWithRetry(collection, {
                $vector: vector,
                text: chunk,
                title,
                release_date,
                rating: movie.vote_average,
                where_to_watch: providers,
                source: `https://www.themoviedb.org/movie/${movie.id}`
              })
              console.log(`✅ Inserted chunk for "${title}"`)
            } catch (err) {
              console.error(`❌ Failed to insert "${title}" chunk:`, err.message)
            }
          }
        })
      )

      await Promise.all(tasks)
      await sleep(300)
    }
  }
}

const main = async () => {
  await createCollection()
  await ingestMovies(2003, 2025)
}

main()
