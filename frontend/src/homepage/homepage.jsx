import React from "react";
import { Link, useNavigate } from "react-router-dom";
import financeImg from "../assets/finance.png";
import logo from "../assets/money.svg";

const Navbar = () => {
  const navigate = useNavigate();

  const handleLogin = () => {
    // Simulate login and navigate to homepage
    navigate("/profile");
  };

  return (
    <nav className="flex justify-between items-center p-4 shadow-md">
      <div className="flex items-center space-x-2">
        <img src={logo} alt="NEPSE Navigator Logo" className="" />
      </div>
      <div className="space-x-6">
        {/* Use Link to navigate instead of <a> */}
        <Link to="/terms" className="text-gray-600 hover:text-customBlue">
          Terms & Conditions
        </Link>
        <Link to="/premium" className="text-gray-600 hover:text-customBlue">
          Premium Plan
        </Link>
        <Link to="/contact" className="text-gray-600 hover:text-customBlue">
          Contact Us
        </Link>
      </div>
      <button onClick={handleLogin} className="w-8 h-8 bg-gray-300 rounded-full flex items-center justify-center">
        <span className="text-white">👤</span>
      </button>
    </nav>
  );
};

const HeroSection = () => {
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

const HomePage = () => {
  return (
    <div>
      <Navbar />
      <HeroSection />
    </div>
  );
};

export default HomePage;
