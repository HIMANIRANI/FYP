// src/premium.jsx
import React from "react";
import { useNavigate } from "react-router-dom";
import logo from "../assets/money.svg";

const Premium = () => {
  const navigate = useNavigate();

  const handleUpgrade = () => {
    // Redirect to the PaymentPage
    navigate("/payment");
  };

  return (
    <div className="flex flex-col items-center p-4 bg-gray-50 min-h-screen">
      {/* Header */}
      <div className="flex justify-between items-center w-full px-6">
        {/* Logo and Title */}
        <div className="flex items-center">
          <img src={logo} alt="Nepse Navigator Logo" className="w-8 h-8 mr-2" />
          <h1 className="text-lg font-bold text-customBlue">NEPSE-Navigator</h1>
        </div>
        {/* Profile */}
        <div className="flex items-center space-x-3">
          <img
            src="/profile-placeholder.png"
            alt="Profile"
            className="w-10 h-10 rounded-free border border-gray-300"
          />
          <span className="text-customBlue font-medium">Your Profile</span>
        </div>
      </div>

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
            <li>• NEPSE rules and regulations</li>
            <li>• Stock Details</li>
            <li>• Finance Knowledge</li>
            <li>• NEPSE-related Bodies Details</li>
          </ul>
          <button
            className="mt-4 bg-gray-300 text-gray-500 cursor-not-allowed py-2 px-6 rounded-lg font-semibold"
            disabled
          >
            Free
          </button>
        </div>

        {/* Premium Plan */}
        <div className="w-[260px] h-[360px] border-2 border-customBlue rounded-2xl p-4 flex flex-col items-center">
          <h2 className="text-3xl font-aboreto mb-2 text-customBlue">PREMIUM</h2>
          <p className="text-xl font-aboreto mb-4 text-customBlue">1000RS/month</p>
          <p className="text-customBlue font-abel text-center mb-4">
            Unlock the Future of Trading: Upgrade to Premium on Nepse Navigator Now
          </p>
          <ul className="text-customBlue font-abel space-y-2">
            <li>• Fundamental Analysis</li>
            <li>• Technical Analysis</li>
            <li>• Stock Comparison</li>
          </ul>
          <button
            className="mt-4 bg-customBlue text-white py-2 px-6 rounded-lg font-semibold hover:bg-blue-700"
            onClick={handleUpgrade}
          >
            Upgrade to Premium
          </button>
        </div>
      </div>
    </div>
  );
};

export default Premium;
