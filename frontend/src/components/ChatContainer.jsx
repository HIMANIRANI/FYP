import { useState, useRef, useEffect } from "react";
import { MessageSquare, Receipt, Plane } from "lucide-react";
import MessageInput from "./MessageInput";
import MessageSkeleton from "../skeletons/MessageSkeleton";

const ChatContainer = () => {
  const messageEndRef = useRef(null);
  const [activeTab, setActiveTab] = useState("chat");

  // Auto-scroll to bottom when messages load (even now it's just placeholder)
  useEffect(() => {
    if (messageEndRef.current) {
      messageEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [activeTab]);

  return (
    <div className="flex flex-col h-full max-h-screen bg-gray-50">

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {activeTab === "chat" && (
          <>
            {/* Chat Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              <MessageSkeleton />
              <div ref={messageEndRef} />
            </div>

            {/* Message Input */}
            <div className="border-t border-base-300 p-4 bg-white">
              <MessageInput />
            </div>
          </>
        )}

        {activeTab === "expenses" && (
          <div className="flex-1 overflow-y-auto p-4">
            <div className="text-center text-gray-500 py-8">
              Personal expenses will be shown here
            </div>
          </div>
        )}

        {activeTab === "trips" && (
          <div className="flex-1 overflow-y-auto p-4">
            <div className="text-center text-gray-500 py-8">
              Personal trips will be shown here
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatContainer;
