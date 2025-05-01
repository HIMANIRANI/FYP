import React from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

export default function SuccessPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const method = searchParams.get("method");
  const data = searchParams.get("data");

  const handleReturnHome = () => {
    navigate("/premium"); 
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50 p-4">
      <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 w-full max-w-md">
        <div className="flex flex-col items-center space-y-4 text-center">
          <div className="text-green-500">
            <svg
              className="w-12 h-12"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M5 13l4 4L19 7"
              />
            </svg>
          </div>
          <h1 className="text-2xl font-bold text-green-600">Payment Successful!</h1>
          <p className="text-gray-600">
            Thank you for your payment. Your transaction has been completed successfully.
          </p>
          {method && (
            <p className="text-gray-600">Payment method: {method}</p>
          )}
          {data && (
            <p className="text-gray-600 text-sm break-all">Transaction data: {data}</p>
          )}
          <button
            onClick={handleReturnHome}
            className="mt-6 w-full bg-green-500 text-white py-3 rounded-lg hover:bg-green-600"
          >
            Return to Home
          </button>
        </div>
      </div>
    </div>
  );
}
