import { NavLink, Outlet } from "react-router-dom";
import clsx from "clsx";

const NAV_ITEMS = [
  { to: "/dashboard", label: "Dashboard" },
  { to: "/upload", label: "Upload" },
  { to: "/documents", label: "Documents" },
];

export default function Layout() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-slate-200 bg-white sticky top-0 z-20">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-brand-600 text-white flex items-center justify-center font-bold">D</div>
            <span className="font-semibold text-slate-900">Document Clean &amp; Reconstruct AI</span>
          </div>
          <nav className="flex items-center gap-1">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  clsx(
                    "px-3 py-2 rounded-md text-sm font-medium transition-colors",
                    isActive ? "bg-brand-50 text-brand-700" : "text-slate-600 hover:bg-slate-100",
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="flex-1 max-w-7xl w-full mx-auto px-6 py-8">
        <Outlet />
      </main>
    </div>
  );
}
