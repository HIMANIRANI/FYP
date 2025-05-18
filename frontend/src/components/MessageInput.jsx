import { useState, useRef, useEffect } from "react";
import { MessageSquare, Loader } from "lucide-react";
import toast from "react-hot-toast";

const MessageInput = ({ onSend, disabled }) => {
  const [message, setMessage] = useState("");
  const inputRef = useRef(null);

  // Auto-focus input on mount
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const trimmedMessage = message.trim();
    if (!trimmedMessage) return;

    try {
      // Clear input immediately for better UX
      setMessage("");
      
      if (onSend) {
        await onSend(trimmedMessage);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      toast.error("Failed to send message");
      // Restore the message if sending failed
      setMessage(trimmedMessage);
    }
  };

  const handleKeyDown = (e) => {
    // Submit on Enter (but not with Shift+Enter)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex items-end gap-2 p-4 bg-white border-t">
      <div className="flex-1 relative">
        <input
          ref={inputRef}
          id="messageInput"
          type="text"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a message..."
          disabled={disabled}
          className={`
            py-2.5 w-full pr-12 outline-none border rounded-lg px-5
            ${disabled ? 'bg-gray-100 cursor-not-allowed' : 'bg-white'}
          `}
        />
      </div>

      <button
        type="submit"
        disabled={!message.trim() || disabled}
        className={`
          btn btn-circle btn-primary
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:bg-primary-dark'}
        `}
        title={disabled ? 'Please wait...' : 'Send message'}
      >
        {disabled ? (
          <Loader className="w-5 h-10 animate-spin" />
        ) : (
          <MessageSquare className="w-7 h-10" />
        )}
      </button>
    </form>
  );
};

export default MessageInput;