import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "@/components/Layout";
import DashboardPage from "@/pages/DashboardPage";
import UploadPage from "@/pages/UploadPage";
import DocumentsListPage from "@/pages/DocumentsListPage";
import DocumentDetailPage from "@/pages/DocumentDetailPage";
import DocumentPreviewPage from "@/pages/DocumentPreviewPage";
import DocumentEditPage from "@/pages/DocumentEditPage";
import DocumentExportPage from "@/pages/DocumentExportPage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/documents" element={<DocumentsListPage />} />
        <Route path="/documents/:id" element={<DocumentDetailPage />} />
        <Route path="/documents/:id/preview" element={<DocumentPreviewPage />} />
        <Route path="/documents/:id/edit" element={<DocumentEditPage />} />
        <Route path="/documents/:id/export" element={<DocumentExportPage />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Route>
    </Routes>
  );
}
