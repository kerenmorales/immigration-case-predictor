import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://iiahbmzrrtgbsmxqifja.supabase.co'
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlpYWhibXpycnRnYnNteHFpZmphIiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njk2ODk4NzgsImV4cCI6MjA4NTI2NTg3OH0.VvWytyChqY_W1K8CQ7PWOr1w_D_2BRHbBVr2YTIsR_U'

export const supabase = createClient(supabaseUrl, supabaseAnonKey)
