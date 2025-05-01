import { GoogleOAuthProvider } from "@react-oauth/google";
import React from "react";
import { Route, Routes } from "react-router-dom";

import GetStarted from "./getStarted/getstarted";
import HomePage from "./homepage/homepage";
import LandingPage from "./landingpage/landingpage";
import Login from "./loginpage/login";
import FailurePage from "./Payment/FailurePage";
import PaymentPage from "./Payment/PaymentPage";
import SuccessPage from "./Payment/SuccessPage";
import Premium from "./premium/premium";
import Profile from "./profile/profile";
import SignupPage from "./signupPage/signupPage";
import TermsAndConditions from "./terms/terms";

function App() {
  return (
    <GoogleOAuthProvider clientId="886481282340-ua5r107135v0lc58kngkgsb0tvvb2kii.apps.googleusercontent.com">
      <Routes>
        <Route path="/" element={<GetStarted />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<SignupPage />} />
        <Route path="/landingpage" element={<LandingPage />} />
        <Route path="/terms" element={<TermsAndConditions />} />
        <Route path="/get-started" element={<GetStarted />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/premium" element={<Premium />} />
        <Route path="/payment" element={<PaymentPage />} />
        <Route path="/failure" element={<FailurePage />} />
        <Route path="/success" element={<SuccessPage />} />
        <Route path="/homepage" element={<HomePage />} />
      </Routes>
    </GoogleOAuthProvider>
  );
}

export default App;
