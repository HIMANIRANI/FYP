import { useState, useRef, useEffect } from "react";
import { MessageSquare } from "lucide-react";
import MessageInput from "./MessageInput";
import MessageSkeleton from "../skeletons/MessageSkeleton";
import { ENDPOINTS, DEFAULT_HEADERS } from "../configreact/api";
import toast from "react-hot-toast";

const ChatContainer = () => {
  const messageEndRef = useRef(null);
  
  const [messages, setMessages] = useState([
    { role: "system", content: "How can I help you?" }
  ]);
  const [loading, setLoading] = useState(false);

  // Auto-scroll to bottom when messages load
  useEffect(() => {
    if (messageEndRef.current) {
      messageEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, loading]);

  // Send message to backend and get response
  const sendMessage = async (userMessage) => {
    try {
      // Add user message to chat
      setMessages((msgs) => [...msgs, { role: "user", content: userMessage }]);
      setLoading(true);

      // Send request to backend
      const response = await fetch(ENDPOINTS.predict, {
        method: "POST",
        headers: DEFAULT_HEADERS,
        body: JSON.stringify({ question: userMessage })
      });

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
      toast.error("Failed to get response from the server");
      setMessages((msgs) => [...msgs, { 
        role: "assistant", 
        content: "Sorry, there was an error processing your request. Please try again." 
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col overflow-y-auto h-full max-h-screen bg-gray-50">
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
                // style={{ display: "block" }}
                className={` break-all rounded-lg px-4 py-2 inline-block ${
                  msg.role === "user"
                    ? "bg-blue-100 ml-auto"
                    : msg.role === "assistant"
                    ? "bg-white border border-gray-200 "
                    : "bg-gray-100 mx-auto text-gray-700"
                }`}
              >
                {msg.content}
                {console.log(msg)}
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
