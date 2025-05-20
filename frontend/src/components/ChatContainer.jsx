import { useState, useRef, useEffect } from "react";
import { MessageSquare } from "lucide-react";
import { useNavigate } from "react-router-dom";
import MessageInput from "./MessageInput";
import MessageSkeleton from "../skeletons/MessageSkeleton";
import { ENDPOINTS, DEFAULT_HEADERS } from "../configreact/api";
import toast from "react-hot-toast";

const ChatContainer = () => {
  const messageEndRef = useRef(null);
  const navigate = useNavigate();
  
  const [messages, setMessages] = useState([
    { role: "system", content: "How can I help you?" }
  ]);
  const [loading, setLoading] = useState(false);
  const [showPremiumPrompt, setShowPremiumPrompt] = useState(false);

  // Fetch chat history when access_token is available or changes
  useEffect(() => {
    const fetchHistory = async () => {
      const token = localStorage.getItem('access_token');
      if (!token) return; // Wait until token is available
      try {
        const response = await fetch("http://localhost:8000/api/chat/history", {
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });
        if (response.ok) {
          const data = await response.json();
          if (data.messages && data.messages.length > 0) {
            const mapped = data.messages.map(msg => ({
              role: msg.role,
              content: msg.content,
              sender: msg.sender,
              receiver: msg.receiver,
              message_limit: msg.message_limit,
              timestamp: msg.timestamp,
            }));
            setMessages([
              { role: "system", content: "How can I help you?" },
              ...mapped,
            ]);
          } else {
            setMessages([{ role: "system", content: "How can I help you?" }]);
          }
        }
      } catch (err) {
        setMessages([{ role: "system", content: "How can I help you?" }]);
      }
    };
    fetchHistory();
  }, [localStorage.getItem('access_token')]);

  // Auto-scroll to bottom when messages load
  useEffect(() => {
    if (messageEndRef.current) {
      messageEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, loading]);

  const handleUpgradeClick = () => {
    navigate("/premium");
  };

  // Send message to backend and get response
  const sendMessage = async (userMessage) => {
    try {
      // Add user message to chat
      setMessages((msgs) => [...msgs, { role: "user", content: userMessage }]);
      setLoading(true);

      // Send request to backend
      const response = await fetch(ENDPOINTS.predict, {
        method: "POST",
        headers: {
          ...DEFAULT_HEADERS,
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`
        },
        body: JSON.stringify({ question: userMessage })
      });

      if (response.status === 403) {
        const error = await response.json();
        setShowPremiumPrompt(true);
        throw new Error(error.detail);
      }

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Handle streaming response
      const reader = response.body.getReader();
      let assistantMessage = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        // Convert the chunk to text
        const chunk = new TextDecoder().decode(value);
        const lines = chunk.split('\n');

        // Process each line
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const content = line.slice(6).trim();
            if (content === 'END') {
              // End of stream, add the complete message
              if (assistantMessage) {
                setMessages((msgs) => [...msgs, { role: "assistant", content: assistantMessage }]);
              }
              break;
            } else {
              // Accumulate the message content
              assistantMessage += content;
            }
          }
        }
      }
    } catch (error) {
      console.error('Error:', error);
      if (error.message.includes("Message limit reached")) {
        toast.error("Message limit reached. Please upgrade to premium for unlimited messages.");
      } else {
        toast.error("Failed to get response from the server");
        setMessages((msgs) => [...msgs, { 
          role: "assistant", 
          content: "Sorry, there was an error processing your request. Please try again." 
        }]);
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col overflow-y-auto h-full max-h-screen bg-gray-50">
      {/* Premium Upgrade Prompt */}
      {showPremiumPrompt && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-lg shadow-xl max-w-md w-full mx-4">
            <h2 className="text-2xl font-bold text-gray-800 mb-4">Upgrade to Premium</h2>
            <p className="text-gray-600 mb-6">
              You've reached your message limit. Upgrade to premium to get unlimited messages and more features!
            </p>
            <div className="flex justify-end space-x-4">
              <button
                onClick={() => setShowPremiumPrompt(false)}
                className="px-4 py-2 text-gray-600 hover:text-gray-800"
              >
                Maybe Later
              </button>
              <button
                onClick={handleUpgradeClick}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Upgrade Now
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`${
                msg.role === "user"
                  ? "flex"
                  : msg.role === "assistant"
                  ? "w-2/3"
                  : "c"
              }`}
            >
              <div
                className={` break-all rounded-lg px-4 py-2 inline-block ${
                  msg.role === "user"
                    ? "bg-blue-100 ml-auto"
                    : msg.role === "assistant"
                    ? "bg-white border border-gray-200 "
                    : "bg-gray-100 mx-auto text-gray-700"
                }`}
              >
               
                {msg.content}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex items-center space-x-2 text-gray-500">
              <div className="animate-bounce">●</div>
              <div className="animate-bounce delay-100">●</div>
              <div className="animate-bounce delay-200">●</div>
            </div>
          )}
          <div ref={messageEndRef} />
        </div>

        {/* Message Input */}
        <MessageInput
          onSend={sendMessage}
          disabled={loading}
        />
      </div>
    </div>
  );
};

export default ChatContainer;
