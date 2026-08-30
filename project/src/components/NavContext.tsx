import { createContext, useContext, useState, useCallback, type ReactNode } from "react";

export type PageId =
  | "overview"
  | "detector"
  | "redteam"
  | "intelligence"
  | "performance"
  | "explainability"
  | "health";

interface NavContextValue {
  page: PageId;
  navigate: (p: PageId) => void;
  collapsed: boolean;
  setCollapsed: (c: boolean) => void;
}

const NavContext = createContext<NavContextValue | null>(null);

export function NavProvider({ children }: { children: ReactNode }) {
  const [page, setPage] = useState<PageId>("overview");
  const [collapsed, setCollapsed] = useState(false);

  const navigate = useCallback((p: PageId) => {
    setPage(p);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  return (
    <NavContext.Provider value={{ page, navigate, collapsed, setCollapsed }}>
      {children}
    </NavContext.Provider>
  );
}

export function useNav() {
  const ctx = useContext(NavContext);
  if (!ctx) throw new Error("useNav must be used within NavProvider");
  return ctx;
}
