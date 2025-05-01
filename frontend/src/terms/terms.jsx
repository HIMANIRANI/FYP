import React from "react";
import { Link } from "react-router-dom";

function TermsAndConditions() {
  return (
    <div className="min-h-screen flex flex-col">
      {/* Top Bar with Circular Back Arrow */}
      <div className="bg-customBlue text-white p-4 text-center font-bold text-xl relative">
        <Link
          to="/signup"
          className="absolute left-4 top-1/2 transform -translate-y-1/2 flex items-center justify-center w-8 h-8 rounded-full bg-white text-customBlue hover:bg-gray-200"
        >
          &#8592; {/* Left Arrow Unicode */}
        </Link>
        Terms and Conditions
      </div>

      {/* Content Section */}
      <div className="flex-grow p-6">
        <h1 className="text-2xl font-bold mb-4">Terms and Conditions for NEPSE Chatbot</h1>

        <div className="space-y-4">
          <section>
            <h2 className="font-bold">1. General Information</h2>
            <p>
              Welcome to the NEPSE Chatbot ("the Chatbot"). By using the Chatbot, you agree to the following terms and conditions. If you do not agree to these terms, please refrain from using the Chatbot.
            </p>
          </section>

          <section>
            <h2 className="font-bold">2. Nature of Data Provided</h2>
            <p>
              The Chatbot provides stock-related information and data sourced from NEPSE (Nepal Stock Exchange). However, please note the following:
            </p>
            <ul className="list-disc ml-6">
              <li>
                <strong>Delayed Data:</strong> The data provided by the Chatbot is not in real-time. The API used by the Chatbot updates only after the market closes and reflects the closing prices of the trading day.
              </li>
              <li>
                <strong>Informational Purposes Only:</strong> The data and information provided are for informational purposes only and should not be considered as financial advice, real-time market data, or a recommendation to buy, sell, or hold any securities.
              </li>
            </ul>
          </section>

          {/* Remaining Sections */}
          <section>
            <h2 className="font-bold">3. Practice Purpose Only</h2>
            <p>
              The Chatbot is developed solely for practice purposes and is not intended for industrial or commercial use. Any reliance on the Chatbot for professional or financial decisions is strongly discouraged.
            </p>
          </section>

          <section>
            <h2 className="font-bold">4. Disclaimer of Liability</h2>
            <p>
              The Chatbot and its creators are not responsible for:
            </p>
            <ul className="list-disc ml-6">
              <li>Any inaccuracies, delays, or errors in the data provided.</li>
              <li>Financial losses or decisions made based on the information provided by the Chatbot.</li>
              <li>Any issues arising from third-party API or data provider limitations or outages.</li>
              <li>
                Mistakes or errors in the information provided by the Chatbot, as it is not a perfect system and may occasionally provide incorrect or incomplete information.
              </li>
            </ul>
          </section>

          {/* Add other sections as needed */}
          <section>
            <h2 className="font-bold">11. Contact Information</h2>
            <p>
              For questions or concerns regarding the Chatbot or these terms and conditions, please contact us at [Insert Contact Information].
            </p>
          </section>

          <section>
            <h2 className="font-bold">Acknowledgment</h2>
            <p>
              By using the NEPSE Chatbot, you acknowledge that you have read, understood, and agreed to these terms and conditions.
            </p>
          </section>
        </div>
      </div>

      {/* Footer */}
      <div className="bg-gray-800 text-white text-center p-4">
        &copy; {new Date().getFullYear()} Himani Shrestha
      </div>
    </div>
  );
}

export default TermsAndConditions;
