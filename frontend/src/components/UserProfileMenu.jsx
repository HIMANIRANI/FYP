import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import Modal from './Modal';
import { ENDPOINTS, createApiClient } from '<div styleName={} />
<configreact></configreact>/api';
import toast from 'react-hot-toast';

const UserProfileMenu = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [userProfile, setUserProfile] = useState(null);
  const [showLogoutModal, setShowLogoutModal] = useState(false);
  const [loading, setLoading] = useState(true);
  const dropdownRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchUserProfile = async () => {
      const token = localStorage.getItem('access_token');
      if (!token) {
        setLoading(false);
        return;
      }

      try {
        const response = await axios.get(
          ENDPOINTS.profile.get,
          createApiClient(token)
        );
        
        setUserProfile(response.data);
        // Cache the profile
        localStorage.setItem('user_profile', JSON.stringify(response.data));
      } catch (error) {
        console.error('Error fetching user profile:', error);
        
        // Try to get from localStorage as fallback
        const profile = localStorage.getItem('user_profile');
        if (profile) {
          try {
            setUserProfile(JSON.parse(profile));
          } catch (e) {
            console.error('Error parsing profile from localStorage:', e);
            toast.error('Error loading user profile');
          }
        } else if (error.response?.status === 401) {
          // Token expired or invalid
          handleLogout();
          toast.error('Session expired. Please log in again.');
        } else {
          toast.error('Failed to load profile');
        }
      } finally {
        setLoading(false);
      }
    };

    fetchUserProfile();
  }, []);

  useEffect(() => {
    // Add click event listener to close dropdown when clicking outside
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const getInitials = () => {
    if (!userProfile) return '';
    
    const firstName = userProfile.firstName || '';
    const lastName = userProfile.lastName || '';
    
    if (!firstName && !lastName) return '';
    
    return `${firstName.charAt(0)}${lastName.charAt(0)}`.toUpperCase();
  };

  const handleLogout = () => {
    setShowLogoutModal(true);
    setIsOpen(false); // Close the dropdown
  };

  const confirmLogout = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user_profile');
    setShowLogoutModal(false);
    toast.success('Logged out successfully');
    navigate('/login');
  };

  const handleProfileClick = () => {
    setIsOpen(false);
    navigate('/profile');
  };

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`
          w-10 h-10 rounded-full flex items-center justify-center text-base font-medium
          transition-colors duration-200
          ${loading ? 'bg-gray-300' : 'bg-customBlue text-white hover:bg-blue-700'}
        `}
        disabled={loading}
        style={{ 
          boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
          letterSpacing: '0.5px'
        }}
      >
        {loading ? (
          <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white" />
        ) : (
          getInitials() || (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-6 w-6"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
              />
            </svg>
          )
        )}
      </button>

      {isOpen && (
        <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 z-50 border border-gray-200">
          <div className="px-4 py-2 border-b border-gray-200">
            {loading ? (
              <div className="animate-pulse">
                <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                <div className="h-3 bg-gray-200 rounded w-1/2"></div>
              </div>
            ) : (
              <>
                <div className="text-sm font-medium text-gray-900">
                  {userProfile?.firstName} {userProfile?.lastName}
                </div>
                <div className="text-xs text-gray-500">{userProfile?.email}</div>
              </>
            )}
          </div>
          <button
            onClick={handleProfileClick}
            className="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
            disabled={loading}
          >
            Profile
          </button>
          <button
            onClick={handleLogout}
            className="block w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-gray-100"
            disabled={loading}
          >
            Logout
          </button>
        </div>
      )}

      <Modal
        isOpen={showLogoutModal}
        onClose={() => setShowLogoutModal(false)}
        onConfirm={confirmLogout}
        title="Confirm Logout"
        message="Are you sure you want to log out? You will need to log in again to access your account."
        confirmText="Logout"
        cancelText="Cancel"
      />
    </div>
  );
};

export default UserProfileMenu; 