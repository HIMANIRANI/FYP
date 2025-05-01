import axios from "axios";
import React, { useEffect, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";

// Import assets
import PlusIcon from "../assets/addimage.png";
import TrashIcon from "../assets/delete.png";
import PencilIcon from "../assets/edit.png";
import UploadIcon from "../assets/uploadpic.png";

function Profile() {
  const [profileImage, setProfileImage] = useState(null);
  const [userData, setUserData] = useState(null);
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [file, setFile] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  
  const navigate = useNavigate();
  const location = useLocation();
  const token = localStorage.getItem("access_token");
  const from = location.state?.from || "/landingpage";

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
        setProfileImage(user.profile_image || "/api/profile/image/default.jpg");
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

  const handleBackToLanding = () => {
    navigate(from);
  };

  const handleImageUpload = (e) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      if (!selectedFile.type.startsWith("image/")) {
        alert("Please select an image file");
        return;
      }
      setFile(selectedFile);
      const reader = new FileReader();
      reader.onloadend = () => {
        setProfileImage(reader.result);
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleDeletePicture = () => {
    setProfileImage("/api/profile/image/default.jpg");
    setFile(null);
  };

  const handleProfileUpdate = async () => {
    setIsLoading(true);
    const formData = new FormData();
    
    // Only append values that have changed
    if (firstName !== userData?.firstName) formData.append("firstName", firstName);
    if (lastName !== userData?.lastName) formData.append("lastName", lastName);
    if (email !== userData?.email) formData.append("email", email);
    if (password) formData.append("password", password);
    if (file) formData.append("profile_image", file);

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
      setProfileImage(updatedUser.profile_image || "/api/profile/image/default.jpg");
      setPassword(""); // Clear password field after successful update
      setFile(null); // Clear file selection
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
      <div className="flex justify-center items-center h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-customBlue mx-auto mb-4"></div>
          <p className="text-gray-600">Loading your profile...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex justify-center items-center h-screen">
        <div className="text-center">
          <p className="text-red-600 mb-4">{error}</p>
          <div className="space-x-4">
            <button
              onClick={() => navigate("/login", { state: { from: "/profile" } })}
              className="bg-customBlue text-white px-4 py-2 rounded hover:bg-customBlue-light"
            >
              Go to Login
            </button>
            <button
              onClick={handleBackToLanding}
              className="border border-customBlue text-customBlue px-4 py-2 rounded hover:bg-gray-50"
            >
              Back to Landing
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!userData) {
    return <div>Loading...</div>; // Show loading state while fetching user data
  }

  return (
    <div className="w-full max-w-3xl mx-auto p-6">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-2xl font-semibold text-customBlue">Profile Settings</h1>
        <button
          onClick={handleBackToLanding}
          className="text-customBlue hover:text-customBlue-light"
        >
          ← Back to Landing
        </button>
      </div>

      <div className="grid gap-8 md:grid-cols-[240px_1fr]">
        {/* Profile Picture Section */}
        <div className="space-y-4">
          <div className="relative">
            <div className="w-[216px] h-[216px] rounded-full bg-gray-200 relative overflow-hidden">
              {profileImage ? (
                <img
                  src={profileImage}
                  alt="Profile"
                  className="w-full h-full object-cover rounded-full"
                  onError={(e) => {
                    e.target.src = "/api/profile/image/default.jpg";
                  }}
                />
              ) : (
                <div className="flex items-center justify-center h-full">
                  <span className="text-gray-500">No image</span>
                </div>
              )}
              <button
                className="absolute bottom-2 right-2 w-8 h-8 bg-customBlue rounded-full flex items-center justify-center"
                onClick={() => document.getElementById("picture-upload")?.click()}
                disabled={isLoading}
              >
                <img src={PlusIcon} alt="Add" className="w-4 h-4" />
              </button>
            </div>
          </div>

          <button
            className="w-full bg-customBlue hover:bg-customBlue-light text-white py-2 px-4 rounded disabled:opacity-50"
            onClick={() => document.getElementById("picture-upload")?.click()}
            disabled={isLoading}
          >
            <img src={UploadIcon} alt="Upload" className="inline-block w-4 h-4 mr-2" />
            Upload Picture
          </button>
          <input
            type="file"
            id="picture-upload"
            className="hidden"
            accept="image/*"
            onChange={handleImageUpload}
            disabled={isLoading}
          />

          <button
            className="w-full border border-red-500 text-red-500 py-2 px-4 rounded hover:bg-red-50 disabled:opacity-50"
            onClick={handleDeletePicture}
            disabled={isLoading}
          >
            <img src={TrashIcon} alt="Delete" className="inline-block w-4 h-4 mr-2" />
            Delete Picture
          </button>
        </div>

        {/* Form Section */}
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
                <button className="absolute right-3 top-1/2 transform -translate-y-1/2">
                  <img src={PencilIcon} alt="Edit" className="w-4 h-4 text-customBlue" />
                </button>
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
                <button className="absolute right-3 top-1/2 transform -translate-y-1/2">
                  <img src={PencilIcon} alt="Edit" className="w-4 h-4 text-customBlue" />
                </button>
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
              <button className="absolute right-3 top-1/2 transform -translate-y-1/2">
                <img src={PencilIcon} alt="Edit" className="w-4 h-4 text-customBlue" />
              </button>
            </div>
          </div>

          <div className="space-y-2">
            <label htmlFor="password" className="text-customBlue font-medium">
              Password
            </label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter new password"
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
