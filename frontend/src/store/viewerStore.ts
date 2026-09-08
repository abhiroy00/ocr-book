import { create } from "zustand";

interface ViewerState {
  currentPage: number;
  zoom: number;
  syncScroll: boolean;
  showOcrBoxes: boolean;
  showLayoutBoxes: boolean;
  showTableBoxes: boolean;
  setCurrentPage: (page: number) => void;
  setZoom: (zoom: number) => void;
  toggleSyncScroll: () => void;
  toggleOcrBoxes: () => void;
  toggleLayoutBoxes: () => void;
  toggleTableBoxes: () => void;
}

export const useViewerStore = create<ViewerState>((set) => ({
  currentPage: 1,
  zoom: 1,
  syncScroll: true,
  showOcrBoxes: false,
  showLayoutBoxes: true,
  showTableBoxes: true,
  setCurrentPage: (page) => set({ currentPage: page }),
  setZoom: (zoom) => set({ zoom: Math.min(4, Math.max(0.25, zoom)) }),
  toggleSyncScroll: () => set((s) => ({ syncScroll: !s.syncScroll })),
  toggleOcrBoxes: () => set((s) => ({ showOcrBoxes: !s.showOcrBoxes })),
  toggleLayoutBoxes: () => set((s) => ({ showLayoutBoxes: !s.showLayoutBoxes })),
  toggleTableBoxes: () => set((s) => ({ showTableBoxes: !s.showTableBoxes })),
}));
