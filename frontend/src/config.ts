/**
 * Application Configuration
 * Resolves the backend API base URL from the VITE_API_URL environment variable,
 * defaulting to same-origin '/api' for local proxying and containerized deployments.
 */
export const API_BASE_URL = (import.meta.env.VITE_API_URL || '/api').replace(/\/$/, '');
