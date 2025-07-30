import { DataAPIClient } from "@datastax/astra-db-ts"
import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer"
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters"
import { GoogleGenerativeAI } from "@google/generative-ai"
import "dotenv/config"

type SimilarityMetric = "dot_product" | "cosine" | "euclidean"

const {
    ASTRA_DB_NAMESPACE,
    ASTRA_DB_COLLECTION,
    ASTRA_DB_API_ENDPOINT,
    ASTRA_DB_APPLICATION_TOKEN,
    GOOGLE_API_KEY
} = process.env

const urlsToScrape = [
    'https://www.imdb.com/',
    'https://www.rottentomatoes.com/',
    'https://www.metacritic.com/',
    'https://www.boxofficemojo.com/',
    'https://www.fandango.com/',
    'https://www.cinemablend.com/',
    'https://www.thewrap.com/',
    'https://www.indiewire.com/',
    'https://www.hollywoodreporter.com/',
    'https://www.variety.com/',
    'https://www.slashfilm.com/',
    'https://www.cinematoday.jp/',
]

// === Setup Gemini ===
const genAI = new GoogleGenerativeAI(GOOGLE_API_KEY!)
const embeddingModel = genAI.getGenerativeModel({ model: "models/embedding-001" })

async function getGeminiEmbedding(text: string): Promise<number[]> {
    const result = await embeddingModel.embedContent({
        content: { role: "user", parts: [{ text }] },
    });
    return result.embedding.values;
}

// === Setup Astra DB ===
const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN!)
const db = client.db(ASTRA_DB_API_ENDPOINT!, { keyspace: ASTRA_DB_NAMESPACE })

const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 512,
    chunkOverlap: 100
})

// === Scrape & Clean ===
const scrapePage = async (url: string): Promise<string> => {
    const loader = new PuppeteerWebBaseLoader(url, {
        launchOptions: { headless: true },
        gotoOptions: { waitUntil: "domcontentloaded" },
        evaluate: async (page, browser) => {
            const html = await page.evaluate(() => document.body.innerHTML)
            await browser.close()
            return html
        }
    })
    const raw = await loader.scrape()
    return raw.replace(/<[^>]*>?/gm, '') // Strip HTML tags
}

// === Create Collection ===
const createCollection = async (similarityMetric: SimilarityMetric = "cosine") => {
    const res = await db.createCollection(ASTRA_DB_COLLECTION!, {
        vector: {
            dimension: 768, // Gemini returns 768-dimensional vectors
            metric: similarityMetric
        }
    })
    console.log("Collection created:", res)
}

// === Process and Insert Chunks ===
const loadSampleData = async () => {
    const collection = await db.collection(ASTRA_DB_COLLECTION!)
    for await (const url of urlsToScrape) {
        try {
            const content = await scrapePage(url)
            const chunks = await splitter.splitText(content)
            for await (const chunk of chunks) {
                const vector = await getGeminiEmbedding(chunk)
                const result = await collection.insertOne({
                    $vector: vector,
                    text: chunk,
                    source: url
                })
                console.log(`Inserted chunk from ${url}:`, result)
            }
        } catch (err) {
            console.error(`Failed to process ${url}:`, err)
        }
    }
}

// === Main ===
const main = async () => {
    await createCollection()
    await loadSampleData()
}

main()
