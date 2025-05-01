// frontend/src/Payment/PaymentPage.jsx
import React, { useState } from "react";
import { useLocation } from "react-router-dom";
import EsewaPayment from "../components/EsewaPayment";
import KhaltiPayment from "../components/KhaltiPayment";

export default function PaymentPage() {
  const { state } = useLocation();
  const [selectedMethod, setSelectedMethod] = useState(null);

  const paymentData = {
    amount: state?.amount || "100",
    productName: state?.productName || "eSewa Test Product",
    transactionId: state?.transactionId || `ESEWA-${Math.random().toString(36).substr(2, 9)}`,
  };

  const handleProceed = () => {
    if (!selectedMethod) {
      alert("Please select a payment method.");
      return;
    }
    // No navigation needed; just show the form
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50 p-4">
      {selectedMethod ? (
        <>
          {selectedMethod === "esewa" && (
            <EsewaPayment
              amount={paymentData.amount}
              productName={paymentData.productName}
              transactionId={paymentData.transactionId}
            />
          )}
          {selectedMethod === "khalti" && (
            <KhaltiPayment
              amount={paymentData.amount}
              productName={paymentData.productName}
              transactionId={paymentData.transactionId}
            />
          )}
        </>
      ) : (
        <div className="bg-white p-6 rounded-lg shadow-md border border-gray-200 w-full max-w-md">
          <h1 className="text-2xl font-bold mb-4 text-gray-800">Payment Gateway</h1>
          <div className="mb-6">
            <p className="text-gray-600">Choose Payment Method</p>
            <p className="text-gray-500">Select your preferred payment option</p>
          </div>
          <div className="space-y-4">
            <label className="flex items-center space-x-2">
              <input
                type="radio"
                name="paymentMethod"
                value="esewa"
                checked={selectedMethod === "esewa"}
                onChange={() => setSelectedMethod("esewa")}
                className="form-radio h-4 w-4 text-green-500 border-gray-300 focus:ring-green-500"
              />
              <span className="text-green-500">eSewa</span>
            </label>
            <label className="flex items-center space-x-2">
              <input
                type="radio"
                name="paymentMethod"
                value="khalti"
                checked={selectedMethod === "khalti"}
                onChange={() => setSelectedMethod("khalti")}
                className="form-radio h-4 w-4 text-purple-500 border-gray-300 focus:ring-purple-500"
              />
              <span className="text-purple-500">Khalti</span>
            </label>
          </div>
          <button
            onClick={handleProceed}
            className="mt-6 w-full bg-black text-white py-3 rounded-lg hover:bg-gray-800"
            disabled={!selectedMethod}
          >
            {selectedMethod ? `Pay with ${selectedMethod.charAt(0).toUpperCase() + selectedMethod.slice(1)}` : "Select a Payment Method"}
          </button>
        </div>
      )}
    </div>
  );
}
