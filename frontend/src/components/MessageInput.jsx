import { useState } from "react";
import { MessageSquare } from "lucide-react";
import toast from "react-hot-toast";

const MessageInput = ({ onSend }) => {
  const [message, setMessage] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!message.trim()) return;

    try {
      let messageData = { text: message.trim() };

      // Clear input immediately for better UX
      setMessage("");

      if (onSend) {
        onSend(messageData);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      toast.error("Failed to send message");
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex items-end gap-2">
      <div className="flex-1">
        <input
          id="messageInput"
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Type a message..."
          className="input input-bordered w-full"
        />
      </div>

      <div className="flex items-center gap-2">
        <button
          type="submit"
          className="btn btn-circle btn-primary"
          disabled={!message.trim()}
        >
          <MessageSquare className="w-5 h-5" />
        </button>
      </div>
    </form>
  );
};

export default MessageInput;