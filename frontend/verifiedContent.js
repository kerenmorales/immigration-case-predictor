/**
 * Verified Content Service
 *
 * All IRCC information shown to paid users MUST come from this service.
 * Never hardcode fees, processing times, eligibility rules, or form fields
 * directly in components — they go stale and we'd mislead users.
 *
 * Every piece of content has:
 *   - source_url: link to canada.ca where it came from
 *   - verified_by: who approved it
 *   - verified_at: when it was approved
 *   - expires_at: when it must be re-verified (default 90 days)
 *
 * If content is expired or unpublished, this service returns null and the
 * UI shows a "Verifying — please check back" placeholder instead of stale info.
 */

import { supabase } from './supabase'

// In-memory cache so we don't hammer the database
const cache = new Map()
const CACHE_TTL_MS = 5 * 60 * 1000 // 5 minutes

const cacheKey = (...parts) => parts.filter(Boolean).join(':')

const isCached = (key) => {
  const entry = cache.get(key)
  return entry && (Date.now() - entry.at) < CACHE_TTL_MS
}

const setCache = (key, value) => {
  cache.set(key, { value, at: Date.now() })
}

const getCache = (key) => cache.get(key)?.value

/**
 * Get all active application types (for showing the "what do you need?" picker)
 */
export async function getApplicationTypes() {
  const key = cacheKey('app_types')
  if (isCached(key)) return getCache(key)

  const { data, error } = await supabase
    .from('application_types')
    .select('*')
    .eq('is_active', true)
    .order('display_order', { ascending: true })

  if (error) {
    console.error('Failed to load application types:', error)
    return []
  }

  setCache(key, data || [])
  return data || []
}

/**
 * Get a single piece of verified content by application + category + key.
 * Returns null if expired, unpublished, or missing.
 */
export async function getContent(applicationTypeId, categoryId, key) {
  const cKey = cacheKey('content', applicationTypeId, categoryId, key)
  if (isCached(cKey)) return getCache(cKey)

  const { data, error } = await supabase
    .from('verified_content')
    .select('*')
    .eq('application_type_id', applicationTypeId)
    .eq('category_id', categoryId)
    .eq('key', key)
    .eq('is_published', true)
    .gt('expires_at', new Date().toISOString())
    .maybeSingle()

  if (error) {
    console.error(`Failed to load content ${applicationTypeId}/${categoryId}/${key}:`, error)
    return null
  }

  setCache(cKey, data)
  return data
}

/**
 * Get all content for a specific application + category (e.g. all form fields for work permit)
 */
export async function getContentList(applicationTypeId, categoryId) {
  const cKey = cacheKey('content_list', applicationTypeId, categoryId)
  if (isCached(cKey)) return getCache(cKey)

  const { data, error } = await supabase
    .from('verified_content')
    .select('*')
    .eq('application_type_id', applicationTypeId)
    .eq('category_id', categoryId)
    .eq('is_published', true)
    .gt('expires_at', new Date().toISOString())
    .order('key', { ascending: true })

  if (error) {
    console.error(`Failed to load content list ${applicationTypeId}/${categoryId}:`, error)
    return []
  }

  setCache(cKey, data || [])
  return data || []
}

/**
 * Helper: get the localized text for a content row
 */
export function localizedText(content, lang = 'es') {
  if (!content) return null
  return lang === 'en' ? content.content_en : content.content_es
}

/**
 * Helper: build a "verified" stamp object for the UI to display next to facts
 */
export function verificationStamp(content) {
  if (!content) return null
  return {
    verifiedBy: content.verified_by,
    verifiedAt: new Date(content.verified_at),
    sourceUrl: content.source_url,
    expiresAt: new Date(content.expires_at),
    isExpiringSoon:
      (new Date(content.expires_at) - Date.now()) < 14 * 24 * 60 * 60 * 1000,
  }
}

/**
 * Clear the cache (call after manual content updates in admin)
 */
export function clearCache() {
  cache.clear()
}
