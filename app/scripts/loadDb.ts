import { DataAPIClient } from "@datastax/astra-db-ts";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAI } from "@google/generative-ai";
import fetch from "node-fetch";
import pLimit from "p-limit";
import crypto from "crypto";
import "dotenv/config";

const {
  ASTRA_DB_NAMESPACE,
  ASTRA_DB_COLLECTION,
  ASTRA_DB_API_ENDPOINT,
  ASTRA_DB_APPLICATION_TOKEN,
  GOOGLE_API_KEY,
  TMDB_API_KEY,
  START_YEAR,
  END_YEAR,
} = process.env;

const startYear = parseInt(START_YEAR || "2000", 10);
const endYear = parseInt(END_YEAR || "2025", 10);

const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN!);
const db = client.db(ASTRA_DB_API_ENDPOINT!, { keyspace: ASTRA_DB_NAMESPACE! });
const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 512, chunkOverlap: 100 });

const genAI = new GoogleGenerativeAI(GOOGLE_API_KEY!);
const embeddingModel = genAI.getGenerativeModel({ model: "models/embedding-001" });

function hashId(input: string) {
  return crypto.createHash("sha256").update(input).digest("hex");
}

async function getGeminiEmbedding(text: string): Promise<number[]> {
  const result = await embeddingModel.embedContent({
    content: { role: "user", parts: [{ text }] },
  });
  return result.embedding.values;
}

async function fetchMoviesByYear(year: number, page: number): Promise<any[]> {
  const url = `https://api.themoviedb.org/3/discover/movie?api_key=${TMDB_API_KEY}&language=en-US&sort_by=popularity.desc&include_adult=false&include_video=false&primary_release_year=${year}&page=${page}`;
  const res = await fetch(url);
  const data = await res.json();
  return data.results || [];
}

async function fetchWatchProviders(movieId: number): Promise<string[]> {
  const url = `https://api.themoviedb.org/3/movie/${movieId}/watch/providers?api_key=${TMDB_API_KEY}`;
  const res = await fetch(url);
  const data = await res.json();
  const providers = data.results?.US?.flatrate || [];
  return providers.map((p: any) => p.provider_name);
}

async function withRetry<T>(fn: () => Promise<T>, retries = 3): Promise<T> {
  for (let i = 0; i <= retries; i++) {
    try {
      return await fn();
    } catch (err) {
      if (i === retries) throw err;
      console.warn(`⏳ Retry ${i + 1} after error: ${err}`);
      await new Promise((res) => setTimeout(res, 1000 * (i + 1)));
    }
  }
  throw new Error("Unreachable");
}

async function ingestMovies(collection: any) {
  console.log(`🚀 INGESTING MOVIES ${startYear}–${endYear}`);
  const limit = pLimit(5); // Concurrency control

  for (let year = startYear; year <= endYear; year++) {
    console.log(`📅 Year: ${year}`);
    for (let page = 1; page <= 500; page++) {
      console.log(`  🌐 Page ${page}`);
      const movies = await fetchMoviesByYear(year, page);
      if (movies.length === 0) break;

      const tasks = movies.map((movie) =>
        limit(async () => {
          const { title, release_date, vote_average } = movie;
          if (!title || !release_date || vote_average < 7.0) return;

          const overview = movie.overview || "No overview available.";
          const providers = await fetchWatchProviders(movie.id);
          const providerText = providers.length > 0 ? `Available on: ${providers.join(", ")}` : "Availability: Unknown";

          const fullContent = `Title: ${title}\nOverview: ${overview}\nRating: ${vote_average}\nRelease Date: ${release_date}\n${providerText}`;
          const chunks = await splitter.splitText(fullContent);

          const chunkDocs = await Promise.all(
            chunks.map(async (chunk, i) => {
              const vector = await withRetry(() => getGeminiEmbedding(chunk));
              const uniqueId = hashId(`${title}_${release_date}_${i}`);
              return {
                _id: uniqueId,
                $vector: vector,
                text: chunk,
                title,
                release_date,
                rating: vote_average,
                where_to_watch: providers,
                source: `https://www.themoviedb.org/movie/${movie.id}`,
                chunk_index: i,
              };
            })
          );

          try {
            await withRetry(() => collection.insertMany(chunkDocs));
            console.log(`✅ Inserted: "${title}" (${chunks.length} chunks)`);
          } catch (err: any) {
            if (err.message.includes("conflict")) {
              console.warn(`⚠️ Skipped duplicate: "${title}"`);
            } else {
              console.error(`❌ Failed to insert ${title}:`, err);
            }
          }
        })
      );

      await Promise.all(tasks);
    }
  }
}

async function main() {
  try {
    await db.createCollection(ASTRA_DB_COLLECTION!, {
      vector: { dimension: 768, metric: "cosine" },
    });
    console.log("✅ Collection created");
  } catch (e: any) {
    if (e.message.includes("already exists")) {
      console.log("ℹ️ Collection already exists");
    } else {
      throw e;
    }
  }

  const collection = await db.collection(ASTRA_DB_COLLECTION!);
  await ingestMovies(collection);
}

main();
