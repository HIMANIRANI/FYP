import React from "react";
import { useNavigate } from "react-router-dom";
import financeImg from "../assets/finance.png";

const HomePage = () => {
  const navigate = useNavigate();

  return (
    <div className="text-center py-12 px-4">
      <p className="text-sm text-gray-500">
        Nepse Navigator is powered by <strong>TheBloke NeuralChat LLM</strong> model, enabling intelligent insights
      </p>
      <h1 className="text-3xl font-bold mt-2">
        <span className="text-customBlue">NEPSE-Navigator:</span> <span className="text-black">Simplifying Nepali Finance for Every Investor</span>
      </h1>
      <button 
        onClick={() => navigate("/landingpage")}
        className="mt-4 px-6 py-2 border rounded-lg text-customBlue border-customBlue hover:bg-customBlue hover:text-white transition"
      >
        Get Started
      </button>
      <div className="flex justify-center mt-8">
        <img src={financeImg} alt="Finance Illustration" className="w-2/3 md:w-1/2" />
      </div>
    </div>
  );
};

export default HomePage;
