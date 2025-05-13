import { GoogleLogin } from "@react-oauth/google";
import axios from "axios";
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import logo from "../assets/money.svg";

const LoginPage = () => {
  const navigate = useNavigate();

  const [loginData, setLoginData] = useState({
    email: "",
    password: "",
  });
  const [error, setError] = useState("");

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setLoginData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  // Handle manual login submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      console.log("Attempting login with:", loginData);

      const response = await axios.post(
        "http://localhost:8000/auth/login",
        {
          email: loginData.email,
          password: loginData.password,
        },
        {
          withCredentials: true, // ✅ Important fix: send cookies/tokens
        }
      );

      console.log("Login response after post:", response.data);

      if (response.data.access_token) {
        localStorage.setItem("access_token", response.data.access_token);

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

      if (error.response) {
        setError(`Login failed: ${error.response.data?.detail || error.response.data || 'Server error'}`);
      } else if (error.request) {
        setError("No response from server. Please check if the server is running.");
      } else {
        setError(`Error: ${error.message}`);
      }
    }
  };

  // Handle Google login
  const handleGoogleLogin = async (credentialResponse) => {
    if (!credentialResponse.credential) {
      console.error("No Google credential found");
      setError("Google login failed - no credential received");
      return;
    }
    try {
      const response = await axios.post(
        "http://localhost:8000/auth/google-login",
        {
          credential: credentialResponse.credential,
        },
        {
          withCredentials: true, // ✅ Important fix here too
        }
      );

      console.log("Google login response:", response.data);

      if (response.data.access_token) {
        localStorage.setItem("access_token", response.data.access_token);

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
        <img src={logo} alt="NEPSE Navigator Logo" />
        <div className="text-4xl font-bold text-customBlue mb-4 text-center">
          Navigating the Nepal Stock Exchange with Ease
        </div>
      </div>

      {/* Right Section */}
      <div className="w-1/2 bg-customBlue flex flex-col justify-center items-center">
        <div className="text-white text-3xl font-semibold mb-4">Welcome Back!</div>
        <p className="text-white text-sm mb-6">Please Enter Your Details</p>

        {/* Display error */}
        {error && <div className="text-red-500 mb-4">{error}</div>}

        {/* Manual Login Form */}
        <form onSubmit={handleSubmit} className="w-3/4 flex flex-col gap-4">
          <input
            type="email"
            name="email"
            placeholder="Email"
            value={loginData.email}
            onChange={handleInputChange}
            className="w-full px-4 py-2 bg-white text-customBlue rounded-lg outline-none focus:ring-2 focus:ring-yellow-500"
            required
          />
          <input
            type="password"
            name="password"
            placeholder="Password"
            value={loginData.password}
            onChange={handleInputChange}
            className="w-full px-4 py-2 bg-white text-customBlue rounded-lg outline-none focus:ring-2 focus:ring-yellow-500"
            required
          />
          <button
            type="submit"
            className="bg-yellow-500 hover:bg-yellow-600 text-white font-semibold py-2 rounded-lg focus:outline-none focus:ring-2 focus:ring-yellow-400"
          >
            Login
          </button>
        </form>

        <div className="text-white mt-4">Or Login With</div>

        {/* Google Login Button */}
        <GoogleLogin
          onSuccess={handleGoogleLogin}
          onError={() => setError("Google login failed")}
        />
      </div>
    </div>
  );
};

export default LoginPage;
