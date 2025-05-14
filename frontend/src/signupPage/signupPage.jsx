import { GoogleLogin } from "@react-oauth/google"; // Import GoogleLogin
import React, { useState } from "react";
import { Link } from "react-router-dom";

const SignupPage = () => {
  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    email: "",
    password: "",
    confirmPassword: "",
    agreeToTerms: false,
  });

  const handleInputChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData({
      ...formData,
      [name]: type === "checkbox" ? checked : value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (formData.password !== formData.confirmPassword) {
      alert("Passwords do not match!");
      return;
    }

    if (!formData.agreeToTerms) {
      alert("You must agree to the Terms & Conditions.");
      return;
    }

    await handleSignup({
      firstName: formData.firstName,
      lastName: formData.lastName,
      email: formData.email,
      password: formData.password,
      confirmPassword: formData.confirmPassword,
    });
  };

  const handleSignup = async (signupData) => {
    try {
      const response = await fetch("http://127.0.0.1:8000/register", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(signupData),
      });

      const result = await response.json();
      console.log("Backend Response:", result);

      if (response.ok) {
        alert(result.message || "Account created successfully!");
        setFormData({
          firstName: "",
          lastName: "",
          email: "",
          password: "",
          confirmPassword: "",
          agreeToTerms: false,
        });
      } else {
        if (result.detail && typeof result.detail === 'string' && result.detail.includes("already exists")) {
          alert("User already exists. Please try logging in.");
        } else {
          const errorMessage = result.detail 
            ? (Array.isArray(result.detail) 
                ? result.detail.map(err => err.msg).join(", ")
                : result.detail)
            : "Something went wrong!";
          alert(errorMessage);
        }
      }
    } catch (error) {
      console.error("Error during signup:", error);
      alert("Failed to create an account. Please try again later.");
    }
  };

  const handleGoogleLogin = async (response) => {
    console.log("Google Login Response:", response);

    try {
      const payload = JSON.parse(atob(response.credential.split(".")[1]));
      console.log("Decoded User Info:", payload);

      const googleAuthData = {
        firstName: payload.given_name || "GoogleUser",
        lastName: payload.family_name || "User",
        email: payload.email,
        password: "google_oauth_placeholder",
        confirmPassword: "google_oauth_placeholder",
      };

      await handleSignup(googleAuthData);
    } catch (error) {
      console.error("Error posting to backend:", error);
      alert("Failed to log in with Google. Please try again later.");
    }
  };

  return (
    <div className="flex h-screen items-center justify-center bg-customBlue">
      <div className="w-1/2 px-16 py-8 bg-white rounded-lg shadow-md">
        <h2 className="text-3xl font-bold mb-4 text-customBlue">
          Create an Account
        </h2>
        <Link to={"/login"} className="mb-6 text-sm text-gray-600">
          Already Have an Account?{" "}
          <span className="text-customBlue underline">LOG IN</span>
        </Link>
        <form onSubmit={handleSubmit}>
          <div className="flex gap-4 mb-4">
            <input
              type="text"
              name="firstName"
              placeholder="First Name"
              value={formData.firstName}
              onChange={handleInputChange}
              className="w-1/2 rounded-md border border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-customBlue"
              required
            />
            <input
              type="text"
              name="lastName"
              placeholder="Last Name"
              value={formData.lastName}
              onChange={handleInputChange}
              className="w-1/2 rounded-md border border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-customBlue"
              required
            />
          </div>
          <input
            type="email"
            name="email"
            placeholder="Email"
            value={formData.email}
            onChange={handleInputChange}
            className="mb-4 w-full rounded-md border border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-customBlue"
            required
          />
          <input
            type="password"
            name="password"
            placeholder="Password"
            value={formData.password}
            onChange={handleInputChange}
            className="mb-4 w-full rounded-md border border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-customBlue"
            required
          />
          <input
            type="password"
            name="confirmPassword"
            placeholder="Confirm Password"
            value={formData.confirmPassword}
            onChange={handleInputChange}
            className="mb-4 w-full rounded-md border border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-customBlue"
            required
          />
          <div className="flex items-center mb-6">
            <input
              type="checkbox"
              id="terms"
              name="agreeToTerms"
              checked={formData.agreeToTerms}
              onChange={handleInputChange}
              className="mr-2 h-4 w-4 rounded border-gray-300 focus:ring-2 focus:ring-customBlue"
              required
            />
            <label htmlFor="terms" className="text-sm text-gray-600">
              I agree to the{" "}
              <Link
                to="/terms"
                className="text-customBlue underline hover:text-customBlueHover"
                target="_blank"
                rel="noopener noreferrer"
              >
                Terms & Conditions
              </Link>
            </label>
          </div>
          <button
            type="submit"
            className="w-full rounded-md bg-customBlue px-4 py-2 text-white font-bold hover:bg-customBlue focus:outline-none focus:ring-2 focus:ring-customBlue"
          >
            Create Account
          </button>
        </form>

        <div className="mt-6">
          <p className="mb-4 text-center text-gray-600">Or login with</p>
          <div className="flex justify-center">
            <GoogleLogin
              onSuccess={handleGoogleLogin}
              onError={(error) => console.log("Google Login Error:", error)}
              useOneTap
              theme="outline"
              size="large"
              className="w-full rounded-md bg-customBlue px-4 py-2 text-white font-bold hover:bg-customBlue focus:outline-none focus:ring-2 focus:ring-customBlue"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

export default SignupPage;
