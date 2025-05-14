import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import logo from '../assets/money.svg';
import UserProfileMenu from './UserProfileMenu';
import axios from 'axios';

// Add keyframes style
const tickerAnimation = `
  @keyframes ticker {
    0% {
      transform: translateX(0);
    }
    100% {
      transform: translateX(-50%);
    }
  }

  .animate-ticker {
    animation: ticker 300s linear infinite;
    display: flex;
    width: fit-content;
  }

  .ticker-container {
    overflow: hidden;
    white-space: nowrap;
    background: white;
    border-bottom: 1px solid #e5e7eb;
  }
`;

// Stock ticker component
const StockTicker = ({ symbol, name, price, change }) => {
  const isPositive = parseFloat(change) >= 0;
  return (
    <div className="px-4 py-2 border-r border-gray-200 flex items-center whitespace-nowrap">
      <span className="font-medium">{symbol}</span>
      <span className="ml-2">Rs.{price}</span>
      <span className={`ml-2 ${isPositive ? 'text-green-500' : 'text-red-500'}`}>
        ({isPositive ? '+' : ''}{change}%)
      </span>
    </div>
  );
};

const Navbar = ({ className = '' }) => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [stockData, setStockData] = useState([]);
  const location = useLocation();
  const isAuthenticated = !!localStorage.getItem('access_token');

  // Add style to head
  useEffect(() => {
    const style = document.createElement('style');
    style.textContent = tickerAnimation;
    document.head.appendChild(style);
    return () => {
      document.head.removeChild(style);
    };
  }, []);

  // Fetch stock data
  useEffect(() => {
    const fetchStockData = async () => {
      try {
        const response = await axios.get("http://localhost:8000/api/stocks/today");
        if (response.data && response.data.data) {
          const formattedData = response.data.data.map(item => ({
            symbol: `${item.company.code}`,
            name: item.company.name,
            price: item.price.close.toString(),
            change: ((item.price.diff / item.price.prevClose) * 100).toFixed(2)
          }));
          setStockData(formattedData);
        }
      } catch (error) {
        console.error("Error fetching stock data:", error);
      }
    };

    fetchStockData();
  }, []);

  const navLinks = [
    { path: isAuthenticated ? '/homepage' : '/', label: 'Home' },
    { path: '/terms', label: 'Terms & Conditions' },
    { path: '/premium', label: 'Premium Plan' },
    { path: '/contact', label: 'Contact Us' },
  ];

  const isActive = (path) => location.pathname === path;

  return (
    <nav className={`bg-white shadow-md w-full ${className}`}>
      {/* Main Navbar */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16">
        <div className="flex justify-between h-full">
          {/* Left: Logo */}
          <div className="flex items-center flex-shrink-0">
            <Link to={isAuthenticated ? "/homepage" : "/"} className="flex items-center">
              <img src={logo} alt="NEPSE Navigator" className="h-20 w-20" />
            </Link>
          </div>

          {/* Center: Navigation Links (Desktop) */}
          <div className="hidden md:flex items-center justify-center space-x-8 flex-shrink-0">
            {navLinks.map((link) => (
              <Link
                key={link.path}
                to={link.path}
                className={`px-3 py-2 rounded-md text-sm font-medium transition-colors ${
                  isActive(link.path)
                    ? 'text-customBlue bg-blue-50'
                    : 'text-gray-600 hover:text-customBlue hover:bg-blue-50'
                }`}
              >
                {link.label}
              </Link>
            ))}
          </div>

          {/* Right: User Menu */}
          <div className="flex items-center">
            {isAuthenticated ? (
              <UserProfileMenu />
            ) : (
              <div className="hidden md:flex items-center space-x-4">
                <Link
                  to="/login"
                  className="text-gray-600 hover:text-customBlue px-3 py-2 rounded-md text-sm font-medium"
                >
                  Login
                </Link>
                <Link
                  to="/signup"
                  className="bg-customBlue text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-blue-700 transition-colors"
                >
                  Sign Up
                </Link>
              </div>
            )}

            {/* Mobile menu button */}
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="md:hidden ml-4 inline-flex items-center justify-center p-2 rounded-md text-gray-600 hover:text-customBlue hover:bg-blue-50 focus:outline-none"
            >
              <span className="sr-only">Open main menu</span>
              {/* Hamburger icon */}
              <svg
                className={`${isMobileMenuOpen ? 'hidden' : 'block'} h-6 w-6`}
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
              {/* Close icon */}
              <svg
                className={`${isMobileMenuOpen ? 'block' : 'hidden'} h-6 w-6`}
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Stock Ticker */}
      <div className="ticker-container">
        <div className="animate-ticker">
          {/* Double the items to create seamless loop */}
          {[...stockData, ...stockData].map((stock, index) => (
            <StockTicker key={index} {...stock} />
          ))}
        </div>
      </div>

      {/* Mobile menu */}
      <div className={`${isMobileMenuOpen ? 'block' : 'hidden'} md:hidden bg-white`}>
        <div className="px-2 pt-2 pb-3 space-y-1">
          {navLinks.map((link) => (
            <Link
              key={link.path}
              to={link.path}
              className={`block px-3 py-2 rounded-md text-base font-medium ${
                isActive(link.path)
                  ? 'text-customBlue bg-blue-50'
                  : 'text-gray-600 hover:text-customBlue hover:bg-blue-50'
              }`}
              onClick={() => setIsMobileMenuOpen(false)}
            >
              {link.label}
            </Link>
          ))}
          {!isAuthenticated && (
            <>
              <Link
                to="/login"
                className="block px-3 py-2 rounded-md text-base font-medium text-gray-600 hover:text-customBlue hover:bg-blue-50"
                onClick={() => setIsMobileMenuOpen(false)}
              >
                Login
              </Link>
              <Link
                to="/signup"
                className="block px-3 py-2 rounded-md text-base font-medium text-gray-600 hover:text-customBlue hover:bg-blue-50"
                onClick={() => setIsMobileMenuOpen(false)}
              >
                Sign Up
              </Link>
            </>
          )}
        </div>
      </div>
    </nav>
  );
};

export default Navbar; 