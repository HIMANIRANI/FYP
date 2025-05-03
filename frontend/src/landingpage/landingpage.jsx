import axios from "axios";
import React, { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import logo from "../assets/money.svg";

// Configure axios defaults
axios.defaults.baseURL = "http://localhost:8000";

// Add keyframes style
const tickerAnimation = `
  @keyframes ticker {
    0% {
      transform: translateX(0);
    }
    100% {
      transform: translateX(-50%);
    }
  }

  .animate-ticker {
    animation: ticker 300s linear infinite;
    display: flex;
    width: fit-content;
  }

  .ticker-container {
    overflow: hidden;
    white-space: nowrap;
    background: white;
    border-bottom: 1px solid #e5e7eb;
  }
`;

// Stock ticker component
const StockTicker = ({ symbol, name, price, change }) => {
  const isPositive = parseFloat(change) >= 0;
  return (
    <div className="px-4 py-2 border-r border-gray-200 flex items-center whitespace-nowrap">
      <span className="font-medium">{symbol}</span>
      <span className="ml-2">Rs.{price}</span>
      <span className={`ml-2 ${isPositive ? 'text-green-500' : 'text-red-500'}`}>
        ({isPositive ? '+' : ''}{change}%)
      </span>
    </div>
  );
};

// Sidebar navigation item
const NavItem = ({ icon, label, isActive, to }) => (
  <Link
    to={to}
    className={`flex items-center space-x-3 px-4 py-3 rounded-lg ${
      isActive ? 'bg-blue-100 text-blue-700' : 'text-gray-700 hover:bg-gray-100'
    }`}
  >
    <span className="text-xl">{icon}</span>
    <span>{label}</span>
  </Link>
);

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
  const [stockData, setStockData] = useState([]);
  const navigate = useNavigate();

  // Add style to head
  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = tickerAnimation;
    document.head.appendChild(style);
    return () => {
      document.head.removeChild(style);
    };
  }, []);

  // Fetch user data
  useEffect(() => {
    const fetchUserData = async () => {
      const token = localStorage.getItem("access_token");
      console.log("Token:", token);

      if (!token) {
        console.log("No token found in localStorage");
        return;
      }

      try {
        console.log("Making API request to /api/profile/get");
        const response = await axios.get("http://localhost:8000/api/profile/get", {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        console.log("User data received:", response.data);
        
        if (response.data) {
          setUserData(response.data);
        } else {
          console.error("No data received from profile endpoint");
        }
      } catch (error) {
        console.error("Error fetching user data:", error.response || error);
        if (error.response?.status === 401) {
          console.log("Unauthorized access, removing token");
          localStorage.removeItem("access_token");
          navigate("/login");
        }
      }
    };

    fetchUserData();
  }, [navigate]);

  // Debug log for userData updates
  useEffect(() => {
    console.log("Current userData state:", userData);
  }, [userData]);

  // Fetch stock data
  useEffect(() => {
    const fetchStockData = async () => {
      try {
        const response = await axios.get("http://localhost:8000/api/stocks/today");
        if (response.data && response.data.data) {
          const formattedData = response.data.data.map(item => ({
            symbol: `${item.company.code}`,
            name: item.company.name,
            price: item.price.close.toString(),
            change: ((item.price.diff / item.price.prevClose) * 100).toFixed(2)
          }));
          setStockData(formattedData);
        }
      } catch (error) {
        console.error("Error fetching stock data:", error);
      }
    };

    fetchStockData();
  }, []);

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
  }, []); // Empty dependency ensures single initialization, reconnection handled in onclose

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

  // Handle profile click
  const handleProfileClick = () => {
    const token = localStorage.getItem("access_token");
    if (!token) {
      navigate("/login", { state: { from: "/landing" } });
    } else {
      navigate("/profile");
    }1
  };

  // Handle premium upgrade click
  const handlePremiumClick = () => {
    navigate("/premium");
  };

  return (
    <div className="h-screen flex flex-col">
      {/* Top Navigation */}
      <nav className="bg-white shadow-md p-4 flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <img src={logo} alt="NEPSE Navigator" className="" />
        </div>
        <div 
          className="flex items-center cursor-pointer" 
          onClick={handleProfileClick}
        >
          <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
            {userData?.profile_image ? (
              <img 
                src={userData.profile_image} 
                alt="Profile" 
                className="w-full h-full rounded-full object-cover"
                onError={(e) => {
                  console.log("Profile image load error");
                  e.target.src = "/api/profile/image/default.jpg";
                }}
              />
            ) : (
              <span>👤</span>
            )}
          </div>
          <span className="ml-2">
            {userData ? 
              (userData.firstName || userData.lastName ? 
                `${userData.firstName || ''} ${userData.lastName || ''}`.trim() : 
                'User'
              ) : 
              'Guest'
            }
          </span>
        </div>
      </nav>

      {/* Stock Tickers */}
      <div className="ticker-container">
        <div className="animate-ticker">
          {/* Double the items to create seamless loop */}
          {[...stockData, ...stockData].map((stock, index) => (
            <StockTicker key={index} {...stock} />
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex">
        {/* Sidebar */}
        <div className="w-64 bg-white border-r p-4">
          <NavItem icon="🏠" label="Home" isActive={true} to="/landing" />
          <NavItem icon="📊" label="Portfolio Management" to="/portfolio" />
          <NavItem icon="📈" label="Stock Watchlist" to="/watchlist" />
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
