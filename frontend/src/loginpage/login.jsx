import { GoogleLogin } from "@react-oauth/google"; // Import GoogleLogin component
import axios from "axios";
import React, { useState } from "react";
import { useNavigate } from "react-router-dom"; // Import useNavigate from react-router-dom
import logo from "../assets/money.svg";

const LoginPage = () => {
  const navigate = useNavigate(); // Initialize the navigate function

  const [loginData, setLoginData] = useState({
    email: "",
    password: "",
  });
  const [error, setError] = useState("");

  // Handle input changes for login form
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setLoginData({
      ...loginData,
      [name]: value,
    });
  };

  // Handle manual login submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      console.log("Attempting login with:", { email: loginData.email });
      const response = await axios.post("http://localhost:8000/auth/login", {
        email: loginData.email,
        password: loginData.password,
      });
      console.log("Login response:", response.data);
      
      if (response.data.access_token) {
        localStorage.setItem("access_token", response.data.access_token);
        // Store user profile data
        if (response.data.user) {
          localStorage.setItem("user_profile", JSON.stringify(response.data.user));
        }
        navigate("/landingpage");
      } else {
        console.error("No access token in response:", response.data);
        setError("Login failed - no access token received");
      }
    } catch (error) {
      console.error("Login error:", error);
      console.error("Response data:", error.response?.data);
      console.error("Response status:", error.response?.status);
      console.error("Response headers:", error.response?.headers);
      
      if (error.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        setError(`Login failed: ${error.response.data?.detail || error.response.data || 'Server error'}`);
      } else if (error.request) {
        // The request was made but no response was received
        setError("No response from server. Please check if the server is running.");
      } else {
        // Something happened in setting up the request that triggered an Error
        setError(`Error: ${error.message}`);
      }
    }
  };

  // Handle Google login response
  const handleGoogleLogin = async (credentialResponse) => {
    if (!credentialResponse.credential) {
      console.error("No Google credential found");
      setError("Google login failed - no credential received");
      return;
    }
    try {
      const response = await axios.post("http://localhost:8000/auth/google-login", {
        token: credentialResponse.credential,
      });
      console.log("Google login response:", response.data);
      if (response.data.access_token) {
        localStorage.setItem("access_token", response.data.access_token);
        // Store user profile data from Google login
        if (response.data.user) {
          localStorage.setItem("user_profile", JSON.stringify(response.data.user));
        }
        navigate("/landingpage");
      } else {
        console.error("No access token received from Google login");
        setError("Google login failed - no access token received");
      }
    } catch (error) {
      console.error("Google login error:", error.response?.data || error.message);
      setError(error.response?.data?.detail || "Google login failed");
    }
  };

  return (
    <div className="flex h-screen">
      {/* Left Section */}
      <div className="w-1/2 bg-white flex flex-col justify-center items-center">
        <img
          src={logo}
          alt="NEPSE Navigator Logo"
          className=""
        />
        <div className="text-4xl font-bold text-customBlue mb-4 text-center">
          Navigating the Nepal Stock Exchange with Ease
        </div>
      </div>

      {/* Right Section */}
      <div className="w-1/2 bg-customBlue flex flex-col justify-center items-center">
        <div className="text-white text-3xl font-semibold mb-4">Welcome Back!</div>
        <p className="text-white text-sm mb-6">Please Enter Your Details</p>

        {/* Display error if any */}
        {error && <div className="text-red-500 mb-4">{error}</div>}

        <form onSubmit={handleSubmit} className="w-3/4 flex flex-col gap-4">
          {/* Email Input */}
          <input
            type="email"
            name="email"
            placeholder="Email"
            value={loginData.email}
            onChange={handleInputChange}
            className="w-full px-4 py-2 bg-white text-customBlue rounded-lg outline-none focus:ring-2 focus:ring-yellow-500"
            required
          />

          {/* Password Input */}
          <input
            type="password"
            name="password"
            placeholder="Password"
            value={loginData.password}
            onChange={handleInputChange}
            className="w-full px-4 py-2 bg-white text-customBlue rounded-lg outline-none focus:ring-2 focus:ring-yellow-500"
            required
          />

          {/* Login Button */}
          <button
            type="submit"
            className="bg-yellow-500 hover:bg-yellow-600 text-white font-semibold py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
          >
            Login
          </button>
        </form>

        <div className="text-white mt-4">Or Login With</div>

        {/* Google Login */}
        <GoogleLogin
          onSuccess={handleGoogleLogin}
          onError={() => setError("Google login failed")}
        />
      </div>
    </div>
  );
};

export default LoginPage;
