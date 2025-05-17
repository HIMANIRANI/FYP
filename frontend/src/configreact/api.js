// API Configuration
export const API_BASE_URL = 'http://localhost:8000';

// API Endpoints
export const ENDPOINTS = {
    // Chat endpoints
    predict: `${API_BASE_URL}/api/predict`,
    
    // User endpoints
    profile: {
        get: `${API_BASE_URL}/api/profile/get`,
        update: `${API_BASE_URL}/api/profile/update`,
    },
    
    // Data endpoints
    stocks: `${API_BASE_URL}/api/stocks/today`,
    
    // Payment endpoints
    payment: `${API_BASE_URL}/api/initiate-payment`,
};

// API Headers
export const DEFAULT_HEADERS = {
    'Content-Type': 'application/json',
};

// Create axios instance with default config
export const createApiClient = (token = null) => {
    const headers = { ...DEFAULT_HEADERS };
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    return {
        headers,
        withCredentials: true,
    };
}; 