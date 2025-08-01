'use client';

import { useState } from 'react';
import './chat.css';

export default function Home() {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState('');

  const quickPrompts = [
  "Is The Dark Knight worth watching?",
  "Where can I stream Oppenheimer?",
  "Best action movies from the 90s",
  "Similar movies to Inception",
  "Who directed Parasite?",
  "How long is Avatar: The Way of Water?",
  "Top rated sci-fi movies on IMDB",
  "Underrated comedy films from 2020s",
  "Why is Pulp Fiction considered a classic?",
  "Key themes in Get Out"
];

  const handleQuickPrompt = (prompt: string) => {
    setQuestion(prompt);
    handleSubmit(prompt);
  };

  const handleSubmit = async (customPrompt?: string) => {
    const prompt = customPrompt || question;
    setResponse('Thinking...');

    const res = await fetch('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ messages: [{ role: 'user', content: prompt }] }),
    });

    const data = await res.text();
    setResponse(data);
  };

  return (
    <main className="container">
      <h1 className="logo">
        <span className="white">Cine</span>RAG
      </h1>

      <p className="description">
        The Ultimate place for Cinephiles! Ask  anything about your favorite movies and it will come back with the most up-to-date answers. We hope you enjoy!
      </p>

      <div className="quick-prompts">
        {quickPrompts.map((prompt, idx) => (
          <button key={idx} className="prompt-button" onClick={() => handleQuickPrompt(prompt)}>
            {prompt}
          </button>
        ))}
      </div>

      <div className="card">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSubmit();
          }}
          className="input-row"
        >
          <input
            className="chat-input"
            placeholder="Ask me something..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
          <button type="submit" className="submit-button">Submit</button>
        </form>

        {response && (
          <div className="response-box">
            {response}
          </div>
        )}
      </div>
    </main>
  );
}
