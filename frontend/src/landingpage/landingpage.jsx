import axios from "axios";
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import ChatContainer from "../components/ChatContainer";
import UserProfileMenu from "../components/UserProfileMenu";
import Watchlist from "../components/Watchlist";

// Import icons
import { MdHome, MdShowChart, MdPlaylistAddCheck, MdStar } from "react-icons/md";

// Axios base URL setup
axios.defaults.baseURL = "http://localhost:8000";

const LandingPage = () => {
  const navigate = useNavigate();
  const [currentPage, setCurrentPage] = useState("home");

  const handlePremiumClick = () => {
    navigate("/premium");
  };

  const renderContent = () => {
    switch (currentPage) {
      case "home":
        return <ChatContainer />;
      case "portfolio":
        return <div className="text-2xl p-6">Portfolio Management Page 📊</div>;
      case "watchlist":
        return <Watchlist />;
      case "chat":
        return <div className="text-2xl p-6">Chat Section Coming Soon 💬</div>;
      default:
        return <div className="text-2xl p-6">Welcome to Nepse Navigator!</div>;
    }
  };

  return (
    <div className="h-screen flex flex-col">
      {/* Main layout */}
      <div className="flex-1 flex">
        {/* Sidebar */}
        <div className="w-64 bg-white border-r p-4">
          <button
            onClick={() => setCurrentPage("home")}
            className={`flex items-center space-x-3 w-full px-4 py-3 rounded-lg text-gray-700 hover:bg-gray-100 ${currentPage === "home" && "bg-gray-100"}`}
          >
            <MdHome className="text-xl" size={30} />
            <span className="w-full truncate">Home</span>
          </button>
          <button
            onClick={() => setCurrentPage("portfolio")}
            className={`flex items-center space-x-3 w-full px-4 py-3 rounded-lg text-gray-700 hover:bg-gray-100 ${currentPage === "portfolio" && "bg-gray-100"}`}
          >
            <MdShowChart className="text-xl" />
            <span className="w-full truncate">Portfolio Management</span>
          </button>
          <button
            onClick={() => setCurrentPage("watchlist")}
            className={`flex items-center space-x-3 w-full px-4 py-3 rounded-lg text-gray-700 hover:bg-gray-100 ${currentPage === "watchlist" && "bg-gray-100"}`}
          >
            <MdPlaylistAddCheck className="text-xl" />
            <span className="w-full truncate">Stock Watchlist</span>
          </button>

          {/* Premium Upgrade Box */}
          <div
            className="mt-4 p-4 bg-yellow-50 rounded-lg cursor-pointer hover:bg-yellow-100 transition-colors"
            onClick={handlePremiumClick}
          >
            <div className="flex items-center text-yellow-600">
              <MdStar className="text-xl mr-2" />
              <span className="w-full truncate">Upgrade to Premium</span>
            </div>
          </div>
        </div>

        {/* Main Page Content */}
        <div className="flex-1 flex flex-col bg-gray-50">
          {renderContent()}
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
