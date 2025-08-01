import { GoogleGenerativeAI } from "@google/generative-ai";
import { DataAPIClient } from "@datastax/astra-db-ts";

const {
  ASTRA_DB_NAMESPACE,
  ASTRA_DB_COLLECTION,
  ASTRA_DB_API_ENDPOINT,
  ASTRA_DB_APPLICATION_TOKEN,
  GOOGLE_API_KEY,
} = process.env;

// Gemini setup
const genAI = new GoogleGenerativeAI(GOOGLE_API_KEY!);
const embeddingModel = genAI.getGenerativeModel({ model: "models/embedding-001" });
const chatModel = genAI.getGenerativeModel({ model: "gemini-pro" });

// Astra setup
const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN!);
const db = client.db(ASTRA_DB_API_ENDPOINT!, { keyspace: ASTRA_DB_NAMESPACE! });

export async function POST(req: Request) {
  try {
    const { messages } = await req.json();
    const latestMessage = messages[messages?.length - 1]?.content;

    let docContext = "";

    // Get embedding from Gemini
    const embeddingResponse = await embeddingModel.embedContent({
      content: { role: "user", parts: [{ text: latestMessage }] },
    });
    const embedding = embeddingResponse.embedding.values;

    // Retrieve top-matching vectors from Astra
    try {
      const collection = await db.collection(ASTRA_DB_COLLECTION!);
      const cursor = collection.find(null, {
        sort: {
          $vector: embedding,
        },
        limit: 10,
      });
      const documents = await cursor.toArray();
      const docsMap = documents?.map((doc) => doc.text);
      docContext = JSON.stringify(docsMap);
    } catch (error) {
      console.log("Error in vector search:", error);
    }

    // Original prompt untouched
    const template = {
      role: "system",
      content: `You are an AI assistant who knows everything about Movies.Use the Below Context to augment what you know about movies>The context will provide you with the most recent page data from TMDB,the official movie database and others.
If the context doesn't include the information you need to answer based on your existing knowledge and don't mention the source of your information or what the context does or doesn't include.
Format responses using markdown where appliable and don't return images.
-----------------
START OF CONTEXT
${docContext}
-----------------
END OF CONTEXT 
-------------
QUESTION:${latestMessage}
-----------------------
'`,
    };

    const result = await chatModel.generateContent({
      contents: [template, ...messages],
    });

    const text = result.response.text();
    return new Response(text, {
      headers: { "Content-Type": "text/plain" },
    });
  } catch (error) {
    console.error("Error in POST request:", error);
    return new Response("Internal Server Error", { status: 500 });
  }
}
