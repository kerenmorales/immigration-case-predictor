-- ============================================================
-- Verified Content Seed — Work Permit (IMM 1295) + Spousal Sponsorship
-- ============================================================
-- This is a STARTING POINT. You (Keren) MUST review every row,
-- verify against canada.ca, and update verified_by/source_url
-- before going live. The expires_at is set to 90 days; re-verify before then.
--
-- Run AFTER 001_verified_content.sql.
-- ============================================================

do $$
declare
  verifier text := 'Keren Morales';
  verified_now timestamptz := now();
  expires timestamptz := now() + interval '90 days';
begin

-- ============================================================
-- WORK PERMIT (IMM 1295) — Form fields
-- ============================================================
insert into verified_content (application_type_id, category_id, key, content_en, content_es, metadata, source_url, source_quote, verified_by, verified_at, expires_at) values
('work_permit', 'form_field', 'family_name',
  'Family Name (Surname) — as shown on passport',
  'Apellido(s) — exactamente como aparece en su pasaporte. Use MAYÚSCULAS.',
  '{"required": true, "format": "uppercase", "example": "GARCIA"}',
  'https://www.canada.ca/content/dam/ircc/migration/ircc/english/pdf/kits/forms/imm1295e.pdf',
  'Family name (last name) as shown on passport',
  verifier, verified_now, expires),

('work_permit', 'form_field', 'given_names',
  'Given Name(s) — first and middle names',
  'Nombre(s) de pila — incluya todos los nombres exactamente como aparecen en su pasaporte.',
  '{"required": true, "example": "Maria Elena"}',
  'https://www.canada.ca/content/dam/ircc/migration/ircc/english/pdf/kits/forms/imm1295e.pdf',
  'Given name(s)',
  verifier, verified_now, expires),

('work_permit', 'form_field', 'date_of_birth',
  'Date of Birth — YYYY-MM-DD',
  'Fecha de Nacimiento — formato AAAA-MM-DD (año-mes-día). Por ejemplo: 1985-03-15.',
  '{"required": true, "format": "YYYY-MM-DD"}',
  'https://www.canada.ca/content/dam/ircc/migration/ircc/english/pdf/kits/forms/imm1295e.pdf',
  'Date of birth (YYYY-MM-DD)',
  verifier, verified_now, expires),

('work_permit', 'form_field', 'country_of_birth',
  'Country of Birth',
  'País de Nacimiento — use el nombre actual del país en inglés (ej: Mexico, Colombia, Guatemala).',
  '{"required": true}',
  'https://www.canada.ca/content/dam/ircc/migration/ircc/english/pdf/kits/forms/imm1295e.pdf',
  'Country of birth',
  verifier, verified_now, expires),

('work_permit', 'form_field', 'citizenship',
  'Country of Citizenship — per passport',
  'País de Ciudadanía — el país que emitió su pasaporte. Si tiene doble ciudadanía, indique la del pasaporte que usará para viajar.',
  '{"required": true}',
  'https://www.canada.ca/content/dam/ircc/migration/ircc/english/pdf/kits/forms/imm1295e.pdf',
  'Country of citizenship',
  verifier, verified_now, expires),

('work_permit', 'form_field', 'current_country_of_residence',
  'Country of Current Residence',
  'País de Residencia Actual — donde vive ahora, no necesariamente su país de ciudadanía.',
  '{"required": true}',
  'https://www.canada.ca/content/dam/ircc/migration/ircc/english/pdf/kits/forms/imm1295e.pdf',
  'Country of current residence',
  verifier, verified_now, expires),

('work_permit', 'form_field', 'passport_number',
  'Passport Number',
  'Número de Pasaporte — copie exactamente como aparece. El pasaporte debe ser válido por al menos 6 meses más allá del periodo del permiso.',
  '{"required": true, "warning": "Passport must be valid for the entire work permit period"}',
  'https://www.canada.ca/content/dam/ircc/migration/ircc/english/pdf/kits/forms/imm1295e.pdf',
  'Passport number',
  verifier, verified_now, expires),

('work_permit', 'form_field', 'passport_issue_date',
  'Passport Issue Date — YYYY-MM-DD',
  'Fecha de Emisión del Pasaporte — formato AAAA-MM-DD.',
  '{"required": true, "format": "YYYY-MM-DD"}',
  'https://www.canada.ca/content/dam/ircc/migration/ircc/english/pdf/kits/forms/imm1295e.pdf',
  'Passport issue date',
  verifier, verified_now, expires),

('work_permit', 'form_field', 'passport_expiry_date',
  'Passport Expiry Date — YYYY-MM-DD',
  'Fecha de Vencimiento del Pasaporte — debe ser válido por al menos 6 meses después del periodo del permiso de trabajo.',
  '{"required": true, "format": "YYYY-MM-DD", "warning": "Must be valid for at least 6 months beyond work permit period"}',
  'https://www.canada.ca/content/dam/ircc/migration/ircc/english/pdf/kits/forms/imm1295e.pdf',
  'Passport expiry date',
  verifier, verified_now, expires),

('work_permit', 'form_field', 'employer_name',
  'Employer Name in Canada',
  'Nombre del Empleador en Canadá — el nombre legal de la empresa que lo contrata.',
  '{"required": true}',
  'https://www.canada.ca/content/dam/ircc/migration/ircc/english/pdf/kits/forms/imm1295e.pdf',
  'Name of employer in Canada',
  verifier, verified_now, expires),

('work_permit', 'form_field', 'employer_address',
  'Employer Address in Canada',
  'Dirección del Empleador — dirección completa del lugar de trabajo en Canadá.',
  '{"required": true}',
  'https://www.canada.ca/content/dam/ircc/migration/ircc/english/pdf/kits/forms/imm1295e.pdf',
  'Address of employer',
  verifier, verified_now, expires),

('work_permit', 'form_field', 'job_title',
  'Job Title / Occupation',
  'Puesto / Ocupación — el título oficial del trabajo según su oferta.',
  '{"required": true}',
  'https://www.canada.ca/content/dam/ircc/migration/ircc/english/pdf/kits/forms/imm1295e.pdf',
  'Job title',
  verifier, verified_now, expires),

('work_permit', 'form_field', 'lmia_number',
  'LMIA Number (if applicable)',
  'Número de LMIA — solo si su trabajo requiere LMIA. Si está exento, marque la sección correspondiente.',
  '{"required": false, "conditional": "Only if LMIA required"}',
  'https://www.canada.ca/content/dam/ircc/migration/ircc/english/pdf/kits/forms/imm1295e.pdf',
  'Labour Market Impact Assessment number',
  verifier, verified_now, expires);

-- WORK PERMIT — Fees (verify these against current canada.ca!)
insert into verified_content (application_type_id, category_id, key, content_en, content_es, metadata, source_url, verified_by, verified_at, expires_at) values
('work_permit', 'fee', 'application_fee',
  'Work permit processing fee — please verify current amount on canada.ca',
  'Tarifa de procesamiento del permiso de trabajo — por favor verifique el monto actual en canada.ca',
  '{"amount_cad": null, "verify": true, "note": "Replace with current amount before publishing"}',
  'https://www.canada.ca/en/immigration-refugees-citizenship/services/fees/fee-list.html',
  verifier, verified_now, expires),

('work_permit', 'fee', 'biometrics_fee',
  'Biometrics fee — please verify current amount on canada.ca',
  'Tarifa de biométricos — por favor verifique el monto actual en canada.ca',
  '{"amount_cad": null, "verify": true}',
  'https://www.canada.ca/en/immigration-refugees-citizenship/services/fees/fee-list.html',
  verifier, verified_now, expires);

-- WORK PERMIT — Required documents
insert into verified_content (application_type_id, category_id, key, content_en, content_es, source_url, verified_by, verified_at, expires_at) values
('work_permit', 'document', 'valid_passport',
  'Valid passport (copy of biographical page)',
  'Pasaporte vigente (copia de la página de datos biográficos)',
  'https://www.canada.ca/en/immigration-refugees-citizenship/services/work-canada/permit/temporary/apply.html',
  verifier, verified_now, expires),
('work_permit', 'document', 'job_offer_letter',
  'Job offer letter from Canadian employer',
  'Carta de oferta de trabajo del empleador canadiense',
  'https://www.canada.ca/en/immigration-refugees-citizenship/services/work-canada/permit/temporary/apply.html',
  verifier, verified_now, expires),
('work_permit', 'document', 'lmia_or_exemption',
  'LMIA or proof of LMIA exemption',
  'LMIA o prueba de exención de LMIA',
  'https://www.canada.ca/en/immigration-refugees-citizenship/services/work-canada/permit/temporary/apply.html',
  verifier, verified_now, expires),
('work_permit', 'document', 'photo',
  'Recent passport-style photo (digital, per IRCC specs)',
  'Foto reciente tipo pasaporte (digital, según especificaciones de IRCC)',
  'https://www.canada.ca/en/immigration-refugees-citizenship/services/application/application-forms-guides/photograph-specifications.html',
  verifier, verified_now, expires),
('work_permit', 'document', 'cv_resume',
  'CV / Resume showing work experience',
  'CV / Currículum mostrando experiencia laboral',
  'https://www.canada.ca/en/immigration-refugees-citizenship/services/work-canada/permit/temporary/apply.html',
  verifier, verified_now, expires),
('work_permit', 'document', 'education_documents',
  'Education credentials (diplomas, transcripts)',
  'Credenciales educativas (diplomas, certificados)',
  'https://www.canada.ca/en/immigration-refugees-citizenship/services/work-canada/permit/temporary/apply.html',
  verifier, verified_now, expires),
('work_permit', 'document', 'proof_of_funds',
  'Proof of funds to support yourself',
  'Prueba de fondos para mantenerse',
  'https://www.canada.ca/en/immigration-refugees-citizenship/services/work-canada/permit/temporary/apply.html',
  verifier, verified_now, expires);

-- WORK PERMIT — Common red flags
insert into verified_content (application_type_id, category_id, key, content_en, content_es, source_url, verified_by, verified_at, expires_at) values
('work_permit', 'warning', 'previous_refusal',
  'Previous immigration refusal — must be disclosed',
  'Rechazo previo de inmigración — DEBE declararse. No declarar una solicitud anterior es motivo de rechazo automático y posible prohibición de 5 años. Si tuvo un rechazo, consulte a un abogado antes de aplicar.',
  'https://www.canada.ca/en/immigration-refugees-citizenship/corporate/publications-manuals/operational-bulletins-manuals/standard-requirements/misrepresentation.html',
  verifier, verified_now, expires),
('work_permit', 'warning', 'criminal_history',
  'Criminal history — affects admissibility',
  'Antecedentes penales — afectan la admisibilidad. Cualquier condena, incluso menores como conducir bajo los efectos del alcohol, debe declararse. Considere asesoría legal antes de aplicar.',
  'https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/inadmissibility.html',
  verifier, verified_now, expires),
('work_permit', 'warning', 'expired_passport',
  'Passport must be valid for entire work period',
  'El pasaporte debe estar vigente durante TODO el periodo del permiso de trabajo. Si vence antes, el permiso solo será válido hasta la fecha de vencimiento del pasaporte.',
  'https://www.canada.ca/en/immigration-refugees-citizenship/services/work-canada/permit/temporary/apply.html',
  verifier, verified_now, expires);

-- ============================================================
-- SPOUSAL SPONSORSHIP — Form fields (IMM 1344, 0008, 5532)
-- ============================================================
insert into verified_content (application_type_id, category_id, key, content_en, content_es, metadata, source_url, verified_by, verified_at, expires_at) values
('spousal_sponsorship', 'form_field', 'sponsor_family_name',
  'IMM 1344 — Sponsor Family Name (in CAPS)',
  'IMM 1344 — Apellido del Patrocinador (en MAYÚSCULAS), exactamente como aparece en su documento de ciudadanía o residencia permanente.',
  '{"form": "IMM 1344", "format": "uppercase", "required": true}',
  'https://www.canada.ca/content/dam/ircc/migration/ircc/english/pdf/kits/forms/imm1344e.pdf',
  'Sponsor family name',
  verifier, verified_now, expires),

('spousal_sponsorship', 'form_field', 'sponsor_given_names',
  'IMM 1344 — Sponsor Given Names',
  'IMM 1344 — Nombre(s) de pila del Patrocinador. Incluya todos los nombres en el orden que aparecen en su documento.',
  '{"form": "IMM 1344", "required": true}',
  'https://www.canada.ca/content/dam/ircc/migration/ircc/english/pdf/kits/forms/imm1344e.pdf',
  'Sponsor given names',
  verifier, verified_now, expires),

('spousal_sponsorship', 'form_field', 'sponsor_status',
  'IMM 1344 — Sponsor Citizenship Status',
  'IMM 1344 — Estatus del Patrocinador en Canadá. Solo ciudadanos canadienses o residentes permanentes pueden patrocinar.',
  '{"form": "IMM 1344", "options": ["Canadian Citizen", "Permanent Resident"]}',
  'https://www.canada.ca/content/dam/ircc/migration/ircc/english/pdf/kits/forms/imm1344e.pdf',
  'Sponsor status in Canada',
  verifier, verified_now, expires),

('spousal_sponsorship', 'form_field', 'applicant_family_name',
  'IMM 0008 — Applicant Family Name (in CAPS)',
  'IMM 0008 — Apellido del Solicitante (en MAYÚSCULAS), exactamente como aparece en su pasaporte. Errores aquí causan rechazo.',
  '{"form": "IMM 0008", "format": "uppercase", "required": true}',
  'https://www.canada.ca/content/dam/ircc/migration/ircc/english/pdf/kits/forms/imm0008enu_2d.pdf',
  'Applicant family name',
  verifier, verified_now, expires),

('spousal_sponsorship', 'form_field', 'applicant_given_names',
  'IMM 0008 — Applicant Given Names',
  'IMM 0008 — Nombre(s) de pila del Solicitante, exactamente como en el pasaporte.',
  '{"form": "IMM 0008", "required": true}',
  'https://www.canada.ca/content/dam/ircc/migration/ircc/english/pdf/kits/forms/imm0008enu_2d.pdf',
  'Applicant given names',
  verifier, verified_now, expires),

('spousal_sponsorship', 'form_field', 'date_married',
  'IMM 5532 — Date of Marriage (YYYY-MM-DD)',
  'IMM 5532 — Fecha de Matrimonio (AAAA-MM-DD). Para uniones de hecho, use la fecha en que comenzaron a vivir juntos.',
  '{"form": "IMM 5532", "format": "YYYY-MM-DD"}',
  'https://www.canada.ca/content/dam/ircc/migration/ircc/english/pdf/kits/forms/imm5532e.pdf',
  'Date of marriage',
  verifier, verified_now, expires),

('spousal_sponsorship', 'form_field', 'place_married',
  'IMM 5532 — Place of Marriage (City, Country)',
  'IMM 5532 — Lugar del Matrimonio (Ciudad, País), exactamente como aparece en su acta de matrimonio.',
  '{"form": "IMM 5532"}',
  'https://www.canada.ca/content/dam/ircc/migration/ircc/english/pdf/kits/forms/imm5532e.pdf',
  'Place of marriage',
  verifier, verified_now, expires),

('spousal_sponsorship', 'form_field', 'how_we_met',
  'IMM 5532 — How and when did you first meet?',
  'IMM 5532 — Cómo y cuándo se conocieron. Sea específico con lugar y fecha. IRCC verifica que ambas versiones (suya y de su pareja) coincidan exactamente.',
  '{"form": "IMM 5532", "warning": "Both partners must give matching answers"}',
  'https://www.canada.ca/content/dam/ircc/migration/ircc/english/pdf/kits/forms/imm5532e.pdf',
  'How did you meet',
  verifier, verified_now, expires),

('spousal_sponsorship', 'form_field', 'relationship_history',
  'IMM 5532 — Relationship History',
  'IMM 5532 — Historia de la Relación. Esta es UNA DE LAS SECCIONES MÁS IMPORTANTES. Incluya fechas específicas de visitas, llamadas frecuentes, viajes juntos, encuentros con familia, etc. IRCC usa esto para evaluar si la relación es genuina.',
  '{"form": "IMM 5532", "critical": true}',
  'https://www.canada.ca/content/dam/ircc/migration/ircc/english/pdf/kits/forms/imm5532e.pdf',
  'Relationship history',
  verifier, verified_now, expires);

-- SPOUSAL SPONSORSHIP — Required documents
insert into verified_content (application_type_id, category_id, key, content_en, content_es, source_url, verified_by, verified_at, expires_at) values
('spousal_sponsorship', 'document', 'marriage_certificate',
  'Marriage certificate (long-form, with translation if not English/French)',
  'Acta de matrimonio (formato largo, con traducción si no está en inglés o francés). La traducción debe ser certificada.',
  'https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/family-sponsorship/spouse-partner-children/apply.html',
  verifier, verified_now, expires),
('spousal_sponsorship', 'document', 'photos',
  '20 photos showing your relationship over time',
  '20 fotos mostrando su relación a través del tiempo. Incluya: ceremonia, viajes juntos, con familia, momentos cotidianos. Anote fecha y lugar al reverso.',
  'https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/family-sponsorship/spouse-partner-children/check-application/photos.html',
  verifier, verified_now, expires),
('spousal_sponsorship', 'document', 'communication_proof',
  'Proof of ongoing communication (chat logs, call history)',
  'Prueba de comunicación constante (historial de chats, llamadas, videollamadas). IRCC busca comunicación regular durante toda la relación.',
  'https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/family-sponsorship/spouse-partner-children/apply.html',
  verifier, verified_now, expires),
('spousal_sponsorship', 'document', 'joint_finances',
  'Evidence of joint finances or shared expenses (if applicable)',
  'Evidencia de finanzas conjuntas o gastos compartidos (cuentas bancarias conjuntas, transferencias, etc.).',
  'https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/family-sponsorship/spouse-partner-children/apply.html',
  verifier, verified_now, expires);

-- SPOUSAL SPONSORSHIP — Common red flags
insert into verified_content (application_type_id, category_id, key, content_en, content_es, source_url, verified_by, verified_at, expires_at) values
('spousal_sponsorship', 'warning', 'inconsistent_stories',
  'Both spouses must give matching stories on how they met, dated, etc.',
  'Ambos cónyuges deben dar respuestas IDÉNTICAS sobre cómo se conocieron, cuándo empezaron a salir, fecha de compromiso, etc. IRCC compara las dos versiones — discrepancias son la causa #1 de rechazo.',
  'https://www.canada.ca/en/immigration-refugees-citizenship/corporate/publications-manuals/operational-bulletins-manuals/permanent-residence/family-class.html',
  verifier, verified_now, expires),
('spousal_sponsorship', 'warning', 'short_relationship',
  'Marriages of less than 1 year receive extra scrutiny',
  'Matrimonios de menos de 1 año reciben mayor escrutinio. IRCC pide más evidencia para demostrar que la relación es genuina, no por conveniencia migratoria.',
  'https://www.canada.ca/en/immigration-refugees-citizenship/corporate/publications-manuals/operational-bulletins-manuals/permanent-residence/family-class.html',
  verifier, verified_now, expires),
('spousal_sponsorship', 'warning', 'sponsor_default',
  'Past sponsorship default makes you ineligible',
  'Si patrocinó a alguien antes y no cumplió con el compromiso financiero (la persona recibió beneficios sociales), no puede patrocinar de nuevo hasta pagar la deuda al gobierno.',
  'https://www.canada.ca/en/immigration-refugees-citizenship/corporate/publications-manuals/operational-bulletins-manuals/permanent-residence/family-class/eligibility/sponsor.html',
  verifier, verified_now, expires);

-- ============================================================
-- Seed monitored sources for canada.ca change detection
-- ============================================================
insert into monitored_sources (url, description, notify_email) values
('https://www.canada.ca/en/immigration-refugees-citizenship/services/fees/fee-list.html', 'IRCC fee schedule', verifier),
('https://www.canada.ca/en/immigration-refugees-citizenship/services/work-canada/permit/temporary/apply.html', 'Work permit application page', verifier),
('https://www.canada.ca/en/immigration-refugees-citizenship/services/immigrate-canada/family-sponsorship/spouse-partner-children/apply.html', 'Spousal sponsorship application page', verifier),
('https://www.canada.ca/en/immigration-refugees-citizenship/services/application/check-processing-times.html', 'Processing times', verifier)
on conflict (url) do nothing;

end $$;
