import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App'

// StrictMode disabled: causes Framer Motion double-mount → black screen on navigation
createRoot(document.getElementById('root')!).render(<App />)
