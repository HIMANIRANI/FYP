import axios from "axios";
import React, { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import logo from "../assets/money.svg";
import UserProfileMenu from "../components/UserProfileMenu";

// Configure axios defaults
axios.defaults.baseURL = "http://localhost:8000";

// Chat message component
const ChatMessage = ({ sender, message, time, isBot }) => (
  <div className={`flex ${isBot ? 'justify-start' : 'justify-end'} mb-4`}>
    <div className="max-w-[70%] bg-white p-3 rounded-lg shadow">
      <div className="flex items-center justify-between mb-2">
        <span className="font-medium">{sender}</span>
        <span className="text-xs text-gray-500">{time}</span>
      </div>
      <p className="text-gray-700">{message}</p>
    </div>
  </div>
);

// Chat history item
const ChatHistoryItem = ({ title, time }) => (
  <div className="flex items-center justify-between p-3 hover:bg-gray-100 cursor-pointer">
    <div>
      <h3 className="font-medium">{title}</h3>
      <p className="text-sm text-gray-500">{time}</p>
    </div>
    <span className="text-gray-400">›</span>
  </div>
);

const LandingPage = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [ws, setWs] = useState(null);
  const [userData, setUserData] = useState(null);
  const navigate = useNavigate();

  // Initialize WebSocket connection with reconnection logic
  useEffect(() => {
    let websocket = new WebSocket("ws://127.0.0.1:8000/ws");

    websocket.onopen = () => {
      console.log("WebSocket connected");
    };

    websocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setMessages((prev) => [...prev, { user: false, text: data.response }]);
        setLoading(false);
      } catch (e) {
        console.error("Invalid message format:", e);
        setMessages((prev) => [...prev, { user: false, text: "Error parsing server response" }]);
        setLoading(false);
      }
    };

    websocket.onerror = (error) => {
      console.error("WebSocket error:", error);
      setMessages((prev) => [...prev, { user: false, text: "WebSocket error occurred" }]);
      setLoading(false);
    };

    websocket.onclose = () => {
      console.log("WebSocket closed. Reconnecting...");
      setMessages((prev) => [...prev, { user: false, text: "WebSocket connection closed. Reconnecting..." }]);
      setLoading(false);
      setTimeout(() => {
        setWs(new WebSocket("ws://127.0.0.1:8000/ws"));
      }, 2000);
    };

    setWs(websocket);

    // Cleanup WebSocket on component unmount
    return () => {
      websocket.close();
    };
  }, []);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    const el = document.getElementById("chat-bottom");
    if (el) el.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Handle sending a new message
  const handleSend = () => {
    if (!input.trim() || !ws || ws.readyState !== WebSocket.OPEN) {
      if (!ws || ws.readyState !== WebSocket.OPEN) {
        setMessages((prev) => [...prev, { user: false, text: "WebSocket not connected" }]);
      }
      return;
    }

    // Append user message to chat
    setMessages((prev) => [...prev, { user: true, text: input }]);
    setLoading(true);

    // Send message to WebSocket server
    ws.send(JSON.stringify({ question: input }));
    setInput("");
  };

  // Handle Enter key press to send message
  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      handleSend();
    }
  };

  const handlePremiumClick = () => {
    navigate("/premium");
  };

  return (
    <div className="h-screen flex flex-col">
      {/* Main Content */}
      <div className="flex-1 flex">
        {/* Sidebar */}
        <div className="w-64 bg-white border-r p-4">
          <Link
            to="/homepage"
            className="flex items-center space-x-3 px-4 py-3 rounded-lg text-gray-700 hover:bg-gray-100"
          >
            <span className="text-xl">🏠</span>
            <span>Home</span>
          </Link>
          <Link
            to="/portfolio"
            className="flex items-center space-x-3 px-4 py-3 rounded-lg text-gray-700 hover:bg-gray-100"
          >
            <span className="text-xl">📊</span>
            <span>Portfolio Management</span>
          </Link>
          <Link
            to="/watchlist"
            className="flex items-center space-x-3 px-4 py-3 rounded-lg text-gray-700 hover:bg-gray-100"
          >
            <span className="text-xl">📈</span>
            <span>Stock Watchlist</span>
          </Link>
          <div 
            className="mt-4 p-4 bg-yellow-50 rounded-lg cursor-pointer hover:bg-yellow-100 transition-colors"
            onClick={handlePremiumClick}
          >
            <div className="flex items-center text-yellow-600">
              <span className="text-xl mr-2">⭐</span>
              <span>Upgrade to Premium</span>
            </div>
          </div>
        </div>

        {/* Chat Area */}
        <div className="flex-1 flex flex-col bg-gray-50 p-4">
          <div className="flex-1 overflow-y-auto">
            {messages.map((msg, index) => (
              <ChatMessage
                key={index}
                sender={msg.user ? "You" : "Nepse Navigator"}
                message={msg.text}
                time={new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                isBot={!msg.user}
              />
            ))}
            {loading && (
              <div className="text-gray-500 italic px-4">Nepse Navigator is typing...</div>
            )}
            <div id="chat-bottom" />
          </div>
          <div className="mt-4">
            <div className="bg-white rounded-lg p-4 shadow-sm">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="How can I help you?"
                className="w-full resize-none border-none focus:outline-none"
                rows="2"
              />
              <div className="flex justify-end mt-2">
                <button 
                  className="bg-blue-700 text-white px-4 py-2 rounded hover:bg-blue-800 transition-colors disabled:opacity-50"
                  onClick={handleSend}
                  disabled={loading || !input.trim()}
                >
                  {loading ? "Sending..." : "Send message"}
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Chat History */}
        <div className="w-72 bg-white border-l">
          <div className="p-4 border-b">
            <h2 className="font-semibold">Chat History</h2>
          </div>
          <div className="overflow-y-auto">
            {messages
              .filter(msg => msg.user)
              .map((msg, i) => (
                <ChatHistoryItem
                  key={i}
                  title={msg.text.substring(0, 30) + (msg.text.length > 30 ? "..." : "")}
                  time={new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                />
              ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
