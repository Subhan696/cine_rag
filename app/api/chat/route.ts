import { GoogleGenerativeAI } from "@google/generative-ai";
import { DataAPIClient } from "@datastax/astra-db-ts";

const {
  ASTRA_DB_NAMESPACE,
  ASTRA_DB_COLLECTION,
  ASTRA_DB_API_ENDPOINT,
  ASTRA_DB_APPLICATION_TOKEN,
  GOOGLE_API_KEY,
} = process.env;

// Initialize Gemini
const genAI = new GoogleGenerativeAI(GOOGLE_API_KEY!);
const embeddingModel = genAI.getGenerativeModel({ model: "models/embedding-001" });

const chatModel = genAI.getGenerativeModel({
  model: "gemini-1.5-flash-latest",
  generationConfig: {
    temperature: 0.4,
    topP: 0.9,
    maxOutputTokens: 1500,
  },
});

// AstraDB setup
const client = new DataAPIClient(ASTRA_DB_APPLICATION_TOKEN!);
const db = client.db(ASTRA_DB_API_ENDPOINT!, { keyspace: ASTRA_DB_NAMESPACE! });

async function findExactMovie(title: string, year?: string) {
  try {
    const collection = await db.collection(ASTRA_DB_COLLECTION!);
    const options = {
      limit: 10,
      projection: {
        title: 1 as const,
        release_date: 1 as const,
        rating: 1 as const,
        overview: 1 as const,
        where_to_watch: 1 as const,
        genres: 1 as const,
        director: 1 as const,
        runtime: 1 as const,
        cast: 1 as const,
      },
    };

    const allMovies = await collection.find({}, options).toArray();
    const filtered = allMovies.filter(
      (movie) =>
        movie.title.toLowerCase().includes(title.toLowerCase()) &&
        (!year || movie.release_date?.includes(year))
    );

    return filtered.slice(0, 1);
  } catch (error) {
    console.error("Exact search error:", error);
    return [];
  }
}

async function findSimilarMovies(embedding: number[]) {
  try {
    const collection = await db.collection(ASTRA_DB_COLLECTION!);
    return await collection
      .find({ "$vector": { "$exists": true } }, {
        sort: { $vector: embedding },
        limit: 5,
        includeSimilarity: true,
        projection: {
          title: 1,
          release_date: 1,
          rating: 1,
          overview: 1,
          where_to_watch: 1,
          genres: 1,
          director: 1,
          runtime: 1,
          cast: 1,
        }
      })
      .toArray();
  } catch (error) {
    console.error("Vector search error:", error);
    return [];
  }
}

function formatMovieResponse(movie: any) {
  const year = movie.release_date ? new Date(movie.release_date).getFullYear() : '';
  const rating = movie.rating ? `⭐ ${movie.rating.toFixed(1)}/10` : '';
  const streaming = movie.where_to_watch?.length
    ? `\n📺 **Where to Watch**: ${movie.where_to_watch.join(', ')}`
    : '';

  return `🎬 **${movie.title}** (${year}) ${rating}\n` +
    `${movie.overview || 'No overview available.'}${streaming}`;
}

async function generateMovieAnalysis(movie: any, query: string) {
  const prompt = `You are a film expert analyzing "${movie.title}" (${movie.release_date?.split('-')[0]}) for a viewer. 

**Query**: "${query}"

**Movie Details**:
- Director: ${movie.director || 'Unknown'}
- Rating: ${movie.rating || '?'}/10
- Genres: ${movie.genres?.join(', ') || 'Unknown'}
- Runtime: ${movie.runtime ? `${Math.floor(movie.runtime / 60)}h ${movie.runtime % 60}m` : 'Unknown'}
${movie.cast?.length ? `- Main Cast: ${movie.cast.slice(0, 3).join(', ')}` : ''}

Provide a 2-3 paragraph analysis that:
1. Gives a brief but insightful review
2. Mentions why it might be worth watching (or not)
3. Notes any awards or significant recognition
4. Maintains an engaging, conversational tone
5. Uses markdown formatting for readability`;

  const result = await chatModel.generateContent({
    contents: [{ role: "user", parts: [{ text: prompt }] }],
  });

  return result.response.text();
}

export async function POST(req: Request) {
  try {
    const { messages } = await req.json();
    const query = messages?.[messages.length - 1]?.content?.trim();

    if (!query) {
      return new Response("No movie query provided.", {
        status: 400,
        headers: { "Content-Type": "text/plain" },
      });
    }

    // Try to extract title and optional year from the prompt
    const titleYearMatch = query.match(/(.+?)\s*(?:\((\d{4})\))?$/i);
    const title = titleYearMatch?.[1]?.trim();
    const year = titleYearMatch?.[2];

    let movies = [];

    if (title) {
      // Try exact match first
      movies = await findExactMovie(title, year);
    }

    if (movies.length === 0) {
      // Fallback to semantic search
      const embeddingResponse = await embeddingModel.embedContent({
        content: { role: "user", parts: [{ text: query }] },
      });

      movies = await findSimilarMovies(embeddingResponse.embedding.values);
    }

    let responseText: string;

    if (movies.length === 0) {
      responseText = `I couldn't find information about "${query}". Could you check the title or ask about another movie?`;
    } else if (movies[0].$similarity < 0.7 && !title) {
      responseText =
        `I found these similar movies:\n\n` +
        movies.map((m) => formatMovieResponse(m)).join('\n\n---\n\n') +
        `\n\nWas one of these the movie you meant?`;
    }
    else {
      const mainMovie = movies[0];
      const similarity = mainMovie?.$similarity ?? 1.0;
      const matchedTitle = mainMovie.title?.toLowerCase() || '';
      const intendedTitle = title?.toLowerCase() || '';

      const isStrongMatch =
        similarity >= 0.75 ||
        (title && matchedTitle.includes(intendedTitle));

      if (!isStrongMatch) {
        // Soft match – don’t auto-analyze
        responseText = `I found this movie which might be similar to "${query}":\n\n` +
          formatMovieResponse(mainMovie) +
          `\n\nWas this the movie you meant? If not, please double-check the title.`;
      } else {
        // Confident match – proceed with full analysis
        responseText =
          (await generateMovieAnalysis(mainMovie, query)) ||
          formatMovieResponse(mainMovie);

        if (movies.length > 1 && !query.toLowerCase().includes('similar')) {
          responseText += `\n\n**You might also enjoy**:\n` +
            movies
              .slice(1, 3)
              .map((m) => `- **${m.title}** (${m.release_date?.split('-')[0]})`)
              .join('\n');
        }
      }
    }


    return new Response(responseText, {
      headers: { "Content-Type": "text/plain" },
    });
  } catch (error) {
    console.error("Error in POST request:", error);
    return new Response(
      "Sorry, I'm having trouble accessing movie information right now. Please try again later.",
      { status: 500, headers: { "Content-Type": "text/plain" } }
    );
  }
}
