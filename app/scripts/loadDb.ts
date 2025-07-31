import { DataAPIClient } from "@datastax/astra-db-ts";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAI } from "@google/generative-ai";
import fetch from "node-fetch";
import pLimit from "p-limit";
import crypto from "crypto";
import "dotenv/config";
import { writeFileSync, existsSync, readFileSync } from "fs";

// === Types ===
interface Movie {
  id: number;
  title: string;
  overview: string;
  release_date: string;
  vote_average: number;
}

interface WatchProvider {
  provider_name: string;
}

interface TMDBWatchProvidersResponse {
  results?: {
    US?: {
      flatrate?: WatchProvider[];
    };
  };
}

interface ChunkDocument {
  _id: string;
  $vector: number[];
  text: string;
  title: string;
  release_date: string;
  rating: number;
  where_to_watch: string[];
  source: string;
  chunk_index: number;
}

// === Configuration ===
const CONFIG = {
  RETRY_LIMIT: 3,
  RETRY_BASE_DELAY_MS: 1000,
  CONCURRENCY_LIMIT: 5,
  TMDB_RATE_LIMIT_DELAY_MS: 350, // TMDB has 40 requests/10 second limit
  MAX_PAGES_PER_YEAR: 500,
  MIN_RATING: 7.0,
  VECTOR_DIMENSION: 768,
  PROGRESS_FILE: "ingest_progress.json",
  BATCH_INSERT_SIZE: 20,
  REGION: "US"
};

// === Validate Environment ===
function validateEnv() {
  const requiredVars = [
    "ASTRA_DB_NAMESPACE",
    "ASTRA_DB_COLLECTION",
    "ASTRA_DB_API_ENDPOINT",
    "ASTRA_DB_APPLICATION_TOKEN",
    "GOOGLE_API_KEY",
    "TMDB_API_KEY"
  ];

  const missingVars = requiredVars.filter(v => !process.env[v]);
  if (missingVars.length > 0) {
    throw new Error(`Missing required environment variables: ${missingVars.join(", ")}`);
  }

  const startYear = parseInt(process.env.START_YEAR || "2000", 10);
  const endYear = parseInt(process.env.END_YEAR || "2025", 10);

  if (isNaN(startYear)) throw new Error("Invalid START_YEAR");
  if (isNaN(endYear)) throw new Error("Invalid END_YEAR");
  if (startYear > endYear) throw new Error("START_YEAR must be <= END_YEAR");

  return { startYear, endYear };
}

// === Progress Tracking ===
interface ProgressState {
  currentYear: number;
  completedYears: number[];
  lastSuccessfulPage: Record<number, number>;
}

function loadProgress(): ProgressState {
  try {
    if (existsSync(CONFIG.PROGRESS_FILE)) {
      return JSON.parse(readFileSync(CONFIG.PROGRESS_FILE, 'utf-8'));
    }
  } catch (err) {
    console.warn("Failed to load progress file, starting fresh");
  }
  return {
    currentYear: validateEnv().startYear,
    completedYears: [],
    lastSuccessfulPage: {}
  };
}

function saveProgress(state: ProgressState) {
  try {
    writeFileSync(CONFIG.PROGRESS_FILE, JSON.stringify(state, null, 2));
  } catch (err) {
    console.error("Failed to save progress:", err);
  }
}

// === Astra Setup ===
const client = new DataAPIClient(process.env.ASTRA_DB_APPLICATION_TOKEN!);
const db = client.db(process.env.ASTRA_DB_API_ENDPOINT!, { 
  keyspace: process.env.ASTRA_DB_NAMESPACE! 
});

// === Text Processing ===
const splitter = new RecursiveCharacterTextSplitter({ 
  chunkSize: 512, 
  chunkOverlap: 100 
});

// === Gemini Setup ===
const genAI = new GoogleGenerativeAI(process.env.GOOGLE_API_KEY!);
const embeddingModel = genAI.getGenerativeModel({ 
  model: "models/embedding-001" 
});

// === Helper Functions ===
async function getGeminiEmbedding(text: string): Promise<number[]> {
  try {
    const result = await embeddingModel.embedContent({
      content: { role: "user", parts: [{ text }] },
    });
    return result.embedding.values;
  } catch (err) {
    console.error("Failed to generate embedding:", err);
    throw err;
  }
}

function hashId(input: string): string {
  return crypto.createHash("sha256").update(input).digest("hex");
}

async function fetchWithRateLimit<T>(url: string): Promise<T> {
  const startTime = Date.now();
  try {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`API request failed with status ${res.status}: ${res.statusText}`);
    }
    return await res.json() as T;
  } finally {
    const elapsed = Date.now() - startTime;
    if (elapsed < CONFIG.TMDB_RATE_LIMIT_DELAY_MS) {
      await new Promise(resolve => 
        setTimeout(resolve, CONFIG.TMDB_RATE_LIMIT_DELAY_MS - elapsed)
      );
    }
  }
}

async function fetchMoviesByYear(year: number, page: number): Promise<Movie[]> {
  const url = `https://api.themoviedb.org/3/discover/movie?api_key=${process.env.TMDB_API_KEY}&language=en-US&sort_by=popularity.desc&include_adult=false&include_video=false&primary_release_year=${year}&page=${page}`;
  const data = await fetchWithRateLimit<{ results?: Movie[] }>(url);
  return data.results || [];
}

async function fetchWatchProviders(movieId: number): Promise<string[]> {
  const url = `https://api.themoviedb.org/3/movie/${movieId}/watch/providers?api_key=${process.env.TMDB_API_KEY}`;
  const data = await fetchWithRateLimit<TMDBWatchProvidersResponse>(url);
  const providers = data.results?.[CONFIG.REGION]?.flatrate || [];
  return providers.map(p => p.provider_name);
}

async function withRetry<T>(
  fn: () => Promise<T>,
  context: string,
  retries = CONFIG.RETRY_LIMIT
): Promise<T> {
  for (let i = 0; i <= retries; i++) {
    try {
      return await fn();
    } catch (err) {
      if (i === retries) {
        console.error(`🚨 Final attempt failed for ${context}:`, err);
        throw err;
      }
      const delay = CONFIG.RETRY_BASE_DELAY_MS * (i + 1);
      console.warn(`⏳ Retry ${i + 1}/${retries} for ${context} in ${delay}ms...`);
      await new Promise(res => setTimeout(res, delay));
    }
  }
  throw new Error("Unreachable");
}
// ... (previous imports and configuration remain the same)

// === Modified Batch Processing ===
async function processMovieBatch(
  collection: any,
  movies: Movie[],
  year: number,
  page: number
): Promise<void> {
  const limit = pLimit(CONFIG.CONCURRENCY_LIMIT);
  const batch: ChunkDocument[] = [];
  let duplicateCount = 0;
  let newCount = 0;

  const processMovie = async (movie: Movie) => {
    try {
      const { title, release_date, vote_average, id, overview } = movie;
      
      if (!title || !release_date || vote_average < CONFIG.MIN_RATING) {
        return;
      }

      const providers = await withRetry(
        () => fetchWatchProviders(id),
        `watch providers for ${title} (${id})`
      );

      const providerText = providers.length > 0 
        ? `Available on: ${providers.join(", ")}` 
        : "Availability: Unknown";

      const fullContent = [
        `Title: ${title}`,
        `Overview: ${overview || "No overview available."}`,
        `Rating: ${vote_average}`,
        `Release Date: ${release_date}`,
        providerText
      ].join("\n");

      const chunks = await splitter.splitText(fullContent);

      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        const docId = hashId(`${title}_${release_date}_${i}`);
        
        // Check for existing document
        const exists = await withRetry(
          () => collection.findOne({ _id: docId }, { projection: { _id: 1 } }),
          `duplicate check for ${docId}`
        );

        if (exists) {
          duplicateCount++;
          continue; // Skip this chunk
        }

        const vector = await withRetry(
          () => getGeminiEmbedding(chunk),
          `embedding for ${title} chunk ${i}`
        );

        batch.push({
          _id: docId,
          $vector: vector,
          text: chunk,
          title,
          release_date,
          rating: vote_average,
          where_to_watch: providers,
          source: `https://www.themoviedb.org/movie/${id}`,
          chunk_index: i
        });
        newCount++;
      }

      console.log(`✅ Processed: "${title}" (${chunks.length} chunks)`);
    } catch (err) {
      console.error(`❌ Failed to process movie ${movie.id}:`, err);
    }
  };

  await Promise.all(movies.map(movie => limit(() => processMovie(movie))));

  // Report duplicates found during checking
  if (duplicateCount > 0) {
    console.log(`⚠️ Skipped ${duplicateCount} duplicate chunks during pre-check`);
  }

  // Insert in batches
  for (let i = 0; i < batch.length; i += CONFIG.BATCH_INSERT_SIZE) {
    const batchSlice = batch.slice(i, i + CONFIG.BATCH_INSERT_SIZE);
    try {
      const result = await collection.insertMany(batchSlice, { ordered: false });
      
      // Handle partial failures in batch insert
      if (result.insertedCount < batchSlice.length) {
        const skipped = batchSlice.length - result.insertedCount;
        console.log(`⚠️ Skipped ${skipped} duplicates in batch insert`);
      }
      
      console.log(`📦 Successfully inserted ${result.insertedCount} chunks`);
    } catch (err) {
      if (err.code === 11000) { // Duplicate key error
        // Fallback to individual inserts with duplicate reporting
        for (const doc of batchSlice) {
          try {
            await collection.insertOne(doc);
            console.log(`➕ Inserted chunk ${doc._id}`);
          } catch (insertErr) {
            if (insertErr.code === 11000) {
              console.log(`⏩ Skipped duplicate chunk ${doc._id}`);
            } else {
              console.error(`❌ Failed to insert document ${doc._id}:`, insertErr);
            }
          }
        }
      } else {
        console.error("Failed to insert batch:", err);
      }
    }
  }

  // Final summary
  console.log(`\n📊 Batch Summary:`);
  console.log(`- New chunks inserted: ${newCount}`);
  console.log(`- Duplicates skipped: ${duplicateCount}`);
}

// ... (rest of the code remains the same)
// === Main Ingest Function ===
async function ingestMovies(collection: any, progress: ProgressState) {
  const { startYear, endYear } = validateEnv();
  console.log(`🚀 INGESTING MOVIES ${startYear}–${endYear}`);

  for (let year = progress.currentYear; year <= endYear; year++) {
    console.log(`\n📅 YEAR ${year}`);
    let page = progress.lastSuccessfulPage[year] ? progress.lastSuccessfulPage[year] + 1 : 1;

    while (page <= CONFIG.MAX_PAGES_PER_YEAR) {
      console.log(`  📄 Page ${page}`);
      
      try {
        const movies = await withRetry(
          () => fetchMoviesByYear(year, page),
          `fetch movies year=${year} page=${page}`
        );

        if (movies.length === 0) {
          console.log(`  🏁 No more movies for ${year}`);
          progress.completedYears.push(year);
          break;
        }

        await processMovieBatch(collection, movies, year, page);
        progress.lastSuccessfulPage[year] = page;
        saveProgress(progress);
        page++;
      } catch (err) {
        console.error(`💥 Critical error processing year ${year} page ${page}:`, err);
        progress.currentYear = year;
        saveProgress(progress);
        throw err; // Stop execution on critical errors
      }
    }

    // Move to next year if completed
    if (page > CONFIG.MAX_PAGES_PER_YEAR || progress.completedYears.includes(year)) {
      progress.currentYear = year + 1;
      delete progress.lastSuccessfulPage[year];
      saveProgress(progress);
    }
  }

  console.log("🎉 Ingest completed successfully!");
}

// === Main Execution ===
async function main() {
  try {
    validateEnv();
    
    const collectionName = process.env.ASTRA_DB_COLLECTION!;
    try {
      await db.createCollection(collectionName, {
        vector: { 
          dimension: CONFIG.VECTOR_DIMENSION, 
          metric: "cosine" 
        }
      });
      console.log("✅ Collection created");
    } catch (err) {
      if (err.message.includes("already exists")) {
        console.log("ℹ️ Collection already exists");
      } else {
        throw err;
      }
    }

    const collection = await db.collection(collectionName);
    const progress = loadProgress();
    await ingestMovies(collection, progress);
  } catch (err) {
    console.error("💀 Fatal error:", err);
    process.exit(1);
  }
}

main();