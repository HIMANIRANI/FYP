import axios from "axios";
import React, { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

// Import assets
import PencilIcon from "../assets/edit.png";

function Profile() {
  const [userData, setUserData] = useState(null);
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [email, setEmail] = useState("");
  const [oldPassword, setOldPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const navigate = useNavigate();
  const location = useLocation();
  const token = localStorage.getItem("access_token");

  useEffect(() => {
    const fetchProfile = async () => {
      if (!token) {
        setError("Please log in to view your profile");
        setIsLoading(false);
        navigate("/login", { state: { from: "/profile" } });
        return;
      }

      try {
        const response = await axios.get("http://localhost:8000/api/profile/get", {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        const user = response.data;
        setUserData(user);
        setFirstName(user.firstName || "");
        setLastName(user.lastName || "");
        setEmail(user.email || "");
        setError(null);
      } catch (error) {
        console.error("Error fetching user profile:", error);
        if (error.response?.status === 401) {
          setError("Session expired. Please log in again");
          localStorage.removeItem("access_token");
          navigate("/login", { state: { from: "/profile" } });
        } else {
          setError("Error loading profile. Please try again later.");
        }
      } finally {
        setIsLoading(false);
      }
    };

    fetchProfile();
  }, [token, navigate]);

  const handleProfileUpdate = async () => {
    setIsLoading(true);
    // Validate password change
    if (oldPassword || newPassword || confirmPassword) {
      if (!oldPassword || !newPassword || !confirmPassword) {
        alert("Please fill all password fields to change your password.");
        setIsLoading(false);
        return;
      }
      if (newPassword !== confirmPassword) {
        alert("New password and confirm password do not match.");
        setIsLoading(false);
        return;
      }
    }
    const formData = new FormData();
    if (firstName !== userData?.firstName) formData.append("firstName", firstName);
    if (lastName !== userData?.lastName) formData.append("lastName", lastName);
    if (email !== userData?.email) formData.append("email", email);
    if (oldPassword && newPassword) {
      formData.append("oldPassword", oldPassword);
      formData.append("newPassword", newPassword);
    }
    try {
      const response = await axios.put("/api/profile/update", formData, {
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "multipart/form-data",
        },
      });
      const updatedUser = response.data;
      setUserData(updatedUser);
      setFirstName(updatedUser.firstName || "");
      setLastName(updatedUser.lastName || "");
      setEmail(updatedUser.email || "");
      setOldPassword("");
      setNewPassword("");
      setConfirmPassword("");
      alert("Profile updated successfully");
    } catch (error) {
      console.error("Error updating profile:", error);
      alert(error.response?.data?.detail || "Error updating profile");
    } finally {
      setIsLoading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 min-h-[calc(100vh-4rem)]">
        <div className="flex justify-center items-center h-full">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-customBlue mx-auto mb-4"></div>
            <p className="text-gray-600">Loading your profile...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 min-h-[calc(100vh-4rem)]">
        <div className="flex justify-center items-center h-full">
          <div className="text-center">
            <p className="text-red-600 mb-4">{error}</p>
            <div className="space-x-4">
              <button
                onClick={() => navigate("/login", { state: { from: "/profile" } })}
                className="bg-customBlue text-white px-4 py-2 rounded hover:bg-customBlue-light"
              >
                Go to Login
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!userData) {
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 min-h-[calc(100vh-4rem)]">
        <div className="flex justify-center items-center h-full">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-customBlue mx-auto mb-4"></div>
            <p className="text-gray-600">Loading...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 min-h-[calc(100vh-4rem)]">
      <div className="max-w-3xl mx-auto">
        <h1 className="text-2xl font-semibold text-customBlue mb-8">Profile Settings</h1>
        <div className="space-y-6">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <label htmlFor="first-name" className="text-customBlue font-medium">
                First Name
              </label>
              <div className="relative">
                <input
                  id="first-name"
                  value={firstName}
                  onChange={(e) => setFirstName(e.target.value)}
                  placeholder={userData?.firstName || "Enter first name"}
                  className="w-full border border-gray-300 rounded px-4 py-2 pr-10 focus:outline-none focus:ring-2 focus:ring-customBlue"
                  disabled={isLoading}
                />
              </div>
            </div>
            <div className="space-y-2">
              <label htmlFor="last-name" className="text-customBlue font-medium">
                Last Name
              </label>
              <div className="relative">
                <input
                  id="last-name"
                  value={lastName}
                  onChange={(e) => setLastName(e.target.value)}
                  placeholder={userData?.lastName || "Enter last name"}
                  className="w-full border border-gray-300 rounded px-4 py-2 pr-10 focus:outline-none focus:ring-2 focus:ring-customBlue"
                  disabled={isLoading}
                />
              </div>
            </div>
          </div>
          <div className="space-y-2">
            <label htmlFor="email" className="text-customBlue font-medium">
              Email
            </label>
            <div className="relative">
              <input
                id="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                type="email"
                placeholder={userData?.email || "Enter email"}
                className="w-full border border-gray-300 rounded px-4 py-2 pr-10 focus:outline-none focus:ring-2 focus:ring-customBlue"
                disabled={isLoading}
              />
            </div>
          </div>
          <div className="space-y-2">
            <label htmlFor="old-password" className="text-customBlue font-medium">
              Old Password
            </label>
            <input
              type="password"
              id="old-password"
              value={oldPassword}
              onChange={(e) => setOldPassword(e.target.value)}
              placeholder="Enter old password"
              className="w-full border border-gray-300 rounded px-4 py-2 pr-10 focus:outline-none focus:ring-2 focus:ring-customBlue"
              disabled={isLoading}
            />
          </div>
          <div className="space-y-2">
            <label htmlFor="new-password" className="text-customBlue font-medium">
              New Password
            </label>
            <input
              type="password"
              id="new-password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              placeholder="Enter new password"
              className="w-full border border-gray-300 rounded px-4 py-2 pr-10 focus:outline-none focus:ring-2 focus:ring-customBlue"
              disabled={isLoading}
            />
          </div>
          <div className="space-y-2">
            <label htmlFor="confirm-password" className="text-customBlue font-medium">
              Confirm New Password
            </label>
            <input
              type="password"
              id="confirm-password"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              placeholder="Confirm new password"
              className="w-full border border-gray-300 rounded px-4 py-2 pr-10 focus:outline-none focus:ring-2 focus:ring-customBlue"
              disabled={isLoading}
            />
          </div>
          <div className="flex justify-end mt-8">
            <button
              className="bg-customBlue hover:bg-customBlue-light text-white py-2 px-4 rounded disabled:opacity-50"
              onClick={handleProfileUpdate}
              disabled={isLoading}
            >
              {isLoading ? "Saving..." : "Save"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Profile;
