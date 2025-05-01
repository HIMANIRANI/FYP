// frontend/src/Failure/FailurePage.jsx
import React from "react";
import { useNavigate } from "react-router-dom";

export default function FailurePage() {
  const navigate = useNavigate();

  const handleReturnHome = () => {
    navigate("/premium");
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100 p-4">
      <h1 className="text-2xl font-bold mb-4 text-red-600">Payment Failed</h1>
      <p className="text-gray-700 mb-6">Your payment could not be completed. Please try again.</p>
      <button
        onClick={handleReturnHome}
        className="bg-red-500 text-white py-2 px-6 rounded-lg hover:bg-red-700"
      >
        Return to Home
      </button>
    </div>
  );
}
