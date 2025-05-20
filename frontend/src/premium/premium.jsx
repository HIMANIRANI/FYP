// src/premium.jsx
import React from "react";
import { useNavigate } from "react-router-dom";

const Premium = () => {
  const navigate = useNavigate();

  const handleUpgrade = () => {
    // Redirect to the PaymentPage
    navigate("/payment");
  };

  return (
    <div className="flex flex-col items-center p-4 bg-gray-50 min-h-screen">
      {/* Title */}
      <h1 className="text-4xl font-bold text-center mt-10 mb-6 font-inter text-customBlue">
        Manage your Plan
      </h1>

      {/* Plans */}
      <div className="flex space-x-8">
        {/* Free Plan */}
        <div className="w-[260px] h-[360px] border-2 border-customBlue rounded-2xl p-4 flex flex-col items-center">
          <h2 className="text-3xl font-aboreto mb-2 text-customBlue">FREE</h2>
          <p className="text-xl font-aboreto mb-4 text-customBlue">0RS/month</p>
          <ul className="text-customBlue font-abel space-y-2">
            <li>• Upto two prompts</li>
            <li>• Explore how this chatbot can help you with NEPSE-Navigator</li>
          </ul>
          <button
            className="mt-4 bg-gray-300 text-gray-500 cursor-not-allowed py-2 px-6 rounded-lg font-semibold"
            disabled
          >
            Free
          </button>
        </div>

        {/* Premium Plan */}
        <div className="w-[260px] h-[360px] border-2 border-customBlue rounded-2xl p-4 flex flex-col items-center relative overflow-hidden">
          {/* Popular Badge */}
          <div className="absolute top-4 right-4">
            <span className="bg-yellow-400 text-customBlue px-3 py-1 rounded-full text-sm font-semibold">
              Popular
            </span>
          </div>

          <h2 className="text-3xl font-aboreto mb-2 text-customBlue">PREMIUM</h2>
          <p className="text-xl font-aboreto mb-4 text-customBlue">500RS/month</p>
          <p className="text-customBlue font-abel space-y-2">Unlock the Future of Trading: 
Upgrade to Premium on Nepse Navigator Now</p>
          <ul className="text-customBlue font-abel space-y-2">
            <li>• Unlimited Prompts</li>
            <li>• Portfolio Watchlist</li>
          </ul>
          <button
            onClick={handleUpgrade}
            className="mt-4 bg-customBlue text-white py-2 px-6 rounded-lg font-semibold hover:bg-blue-700 transition-colors"
          >
            Upgrade Now
          </button>
        </div>
      </div>
    </div>
  );
};

export default Premium;
