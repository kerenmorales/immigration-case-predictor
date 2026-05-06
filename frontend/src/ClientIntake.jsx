import { useState, useEffect, useRef } from 'react'
import { supabase } from './supabase'

const t = {
  en: {
    title: 'Client Intake Form',
    subtitle: 'Complete this form before your consultation',
    lang: 'Language / Idioma',
    step1: 'Personal Info',
    step2: 'Immigration Status',
    step3: 'Services Needed',
    step4: 'Education & Work',
    step5: 'Language & Other',
    step6: 'Dependents / Children',
    step7: 'Review & Submit',
    next: 'Next',
    prev: 'Previous',
    submit: 'Submit & Download PDF',
    saving: 'Saving...',
    dependentsTitle: 'Dependents / Children',
    dependentsSubtitle: 'Add all dependent children who will be included in the application',
    addChild: 'Add Child',
    removeChild: 'Remove',
    childName: 'Full Name',
    childDob: 'Date of Birth',
    childGender: 'Gender',
    childRelationship: 'Relationship to You',
    childCitizenship: 'Country of Citizenship',
    childPassportNum: 'Passport Number',
    childPassportExpiry: 'Passport Expiry Date',
    childCountryBirth: 'Country of Birth',
    childMaritalStatus: 'Marital Status (if 18+)',
    childAccompanying: 'Accompanying you to Canada?',
    male: 'Male', female: 'Female', other: 'Other',
    genderOptions: { male: 'Male', female: 'Female', other: 'Other' },
    relationshipOptions: { son: 'Son', daughter: 'Daughter', stepson: 'Stepson', stepdaughter: 'Stepdaughter', adopted_son: 'Adopted Son', adopted_daughter: 'Adopted Daughter' },
    fullName: 'Full Legal Name',
    dob: 'Date of Birth',
    countryBirth: 'Country of Birth',
    citizenship: 'Country of Citizenship',
    currentAddress: 'Current Address',
    phone: 'Phone Number',
    email: 'Email Address',
    preferredLang: 'Preferred Language',
    maritalStatus: 'Marital Status',
    numDependents: 'Number of Dependents',
    single: 'Single', married: 'Married', commonLaw: 'Common-law', divorced: 'Divorced', separated: 'Separated', widowed: 'Widowed',
    currentStatus: 'Current Immigration Status in Canada',
    statusOptions: {
      citizen: 'Canadian Citizen',
      pr: 'Permanent Resident',
      workPermit: 'Work Permit Holder',
      studyPermit: 'Study Permit Holder',
      visitor: 'Visitor Visa',
      superVisa: 'Super Visa',
      refugee: 'Refugee / Asylum Claimant',
      noStatus: 'No Status (Overstayed)',
      outside: 'Outside Canada'
    },
    permitNumber: 'Permit / Visa Number',
    dateIssued: 'Date Issued',
    expiryDate: 'Expiry Date',
    howLongInCanada: 'How long have you been in Canada?',
    everRefused: 'Have you ever been refused entry or had a visa denied?',
    yes: 'Yes', no: 'No',
    refusalDetails: 'If yes, provide details',
    whatLookingToDo: 'What are you looking to do? (select all that apply)',
    serviceOptions: {
      getWorkPermit: 'Get a work permit',
      extendWorkPermit: 'Extend a work permit',
      getPR: 'Get permanent residency',
      sponsorSpouse: 'Sponsor a spouse/partner',
      sponsorParents: 'Sponsor parents/grandparents',
      citizenship: 'Apply for citizenship',
      visitorVisa: 'Get a visitor visa',
      studyPermit: 'Get a study permit',
      refugee: 'Refugee/asylum claim',
      appeal: 'Appeal a refusal',
      other: 'Other'
    },
    otherDetails: 'Please specify',
    education: 'Highest Level of Education',
    eduOptions: {
      none: 'No formal education',
      highSchool: 'High School',
      diploma: 'College Diploma',
      bachelors: "Bachelor's Degree",
      masters: "Master's Degree",
      phd: 'PhD / Doctorate',
      trade: 'Trade Certificate'
    },
    fieldOfStudy: 'Field of Study',
    eduCountry: 'Country Where Education Was Completed',
    wcaAssessed: 'Has education been assessed by WES/ECA?',
    currentOccupation: 'Current Occupation',
    yearsExpCanada: 'Years of Work Experience in Canada',
    yearsExpTotal: 'Total Years of Work Experience',
    nocCode: 'NOC Code (if known)',
    hasJobOffer: 'Do you have a job offer from a Canadian employer?',
    lmiaStatus: 'LMIA Status',
    lmiaOptions: { notApplicable: 'Not Applicable', pending: 'Pending', approved: 'Approved', exempt: 'LMIA Exempt' },
    englishLevel: 'English Proficiency Level',
    frenchLevel: 'French Proficiency Level',
    langLevels: { none: 'None', basic: 'Basic', intermediate: 'Intermediate', advanced: 'Advanced', native: 'Native' },
    ieltsScore: 'IELTS / CELPIP Score (if available)',
    tefScore: 'TEF / TCF Score (if available)',
    familyInCanada: 'Do you have family in Canada?',
    familyDetails: 'Who? (name, relationship, their status)',
    criminalHistory: 'Any criminal history?',
    criminalDetails: 'If yes, provide details',
    medicalIssues: 'Any medical issues that could affect your application?',
    previousApps: 'Previous immigration applications',
    previousAppsDetails: 'List previous applications (type, date, result)',
    budget: 'Budget for immigration services (CAD)',
    additionalNotes: 'Additional Notes or Questions',
    reviewTitle: 'Review Your Information',
    reviewSubtitle: 'Please review before submitting',
    downloadPdf: 'Download PDF',
    saved: 'Client intake saved successfully!',
    required: 'Required'
  },
  es: {
    title: 'Formulario de Admision de Cliente',
    subtitle: 'Complete este formulario antes de su consulta',
    lang: 'Language / Idioma',
    step1: 'Informacion Personal',
    step2: 'Estado Migratorio',
    step3: 'Servicios Necesarios',
    step4: 'Educacion y Trabajo',
    step5: 'Idioma y Otros',
    step6: 'Dependientes / Hijos',
    step7: 'Revisar y Enviar',
    next: 'Siguiente',
    prev: 'Anterior',
    submit: 'Enviar y Descargar PDF',
    saving: 'Guardando...',
    dependentsTitle: 'Dependientes / Hijos',
    dependentsSubtitle: 'Agregue todos los hijos dependientes que seran incluidos en la solicitud',
    addChild: 'Agregar Hijo/a',
    removeChild: 'Eliminar',
    childName: 'Nombre Completo',
    childDob: 'Fecha de Nacimiento',
    childGender: 'Genero',
    childRelationship: 'Relacion con Usted',
    childCitizenship: 'Pais de Ciudadania',
    childPassportNum: 'Numero de Pasaporte',
    childPassportExpiry: 'Fecha de Vencimiento del Pasaporte',
    childCountryBirth: 'Pais de Nacimiento',
    childMaritalStatus: 'Estado Civil (si es mayor de 18)',
    childAccompanying: 'Lo/la acompana a Canada?',
    male: 'Masculino', female: 'Femenino', other: 'Otro',
    genderOptions: { male: 'Masculino', female: 'Femenino', other: 'Otro' },
    relationshipOptions: { son: 'Hijo', daughter: 'Hija', stepson: 'Hijastro', stepdaughter: 'Hijastra', adopted_son: 'Hijo Adoptivo', adopted_daughter: 'Hija Adoptiva' },
    fullName: 'Nombre Legal Completo',
    dob: 'Fecha de Nacimiento',
    countryBirth: 'Pais de Nacimiento',
    citizenship: 'Pais de Ciudadania',
    currentAddress: 'Direccion Actual',
    phone: 'Numero de Telefono',
    email: 'Correo Electronico',
    preferredLang: 'Idioma Preferido',
    maritalStatus: 'Estado Civil',
    numDependents: 'Numero de Dependientes',
    single: 'Soltero/a', married: 'Casado/a', commonLaw: 'Union Libre', divorced: 'Divorciado/a', separated: 'Separado/a', widowed: 'Viudo/a',
    currentStatus: 'Estado migratorio actual en Canada',
    statusOptions: {
      citizen: 'Ciudadano Canadiense',
      pr: 'Residente Permanente',
      workPermit: 'Permiso de Trabajo',
      studyPermit: 'Permiso de Estudio',
      visitor: 'Visa de Visitante',
      superVisa: 'Super Visa',
      refugee: 'Refugiado / Solicitante de Asilo',
      noStatus: 'Sin Estatus (Vencido)',
      outside: 'Fuera de Canada'
    },
    permitNumber: 'Numero de Permiso / Visa',
    dateIssued: 'Fecha de Emision',
    expiryDate: 'Fecha de Vencimiento',
    howLongInCanada: 'Cuanto tiempo lleva en Canada?',
    everRefused: 'Le han negado la entrada o una visa alguna vez?',
    yes: 'Si', no: 'No',
    refusalDetails: 'Si es si, proporcione detalles',
    whatLookingToDo: 'Que necesita hacer? (seleccione todos los que apliquen)',
    serviceOptions: {
      getWorkPermit: 'Obtener permiso de trabajo',
      extendWorkPermit: 'Extender permiso de trabajo',
      getPR: 'Obtener residencia permanente',
      sponsorSpouse: 'Patrocinar esposo/a o pareja',
      sponsorParents: 'Patrocinar padres/abuelos',
      citizenship: 'Solicitar ciudadania',
      visitorVisa: 'Obtener visa de visitante',
      studyPermit: 'Obtener permiso de estudio',
      refugee: 'Solicitud de refugio/asilo',
      appeal: 'Apelar un rechazo',
      other: 'Otro'
    },
    otherDetails: 'Por favor especifique',
    education: 'Nivel mas alto de educacion',
    eduOptions: {
      none: 'Sin educacion formal',
      highSchool: 'Secundaria',
      diploma: 'Diploma Universitario',
      bachelors: 'Licenciatura',
      masters: 'Maestria',
      phd: 'Doctorado',
      trade: 'Certificado Tecnico'
    },
    fieldOfStudy: 'Campo de Estudio',
    eduCountry: 'Pais donde completo su educacion',
    wcaAssessed: 'Su educacion ha sido evaluada por WES/ECA?',
    currentOccupation: 'Ocupacion Actual',
    yearsExpCanada: 'Anos de experiencia laboral en Canada',
    yearsExpTotal: 'Anos totales de experiencia laboral',
    nocCode: 'Codigo NOC (si lo conoce)',
    hasJobOffer: 'Tiene una oferta de trabajo de un empleador canadiense?',
    lmiaStatus: 'Estado de LMIA',
    lmiaOptions: { notApplicable: 'No Aplica', pending: 'Pendiente', approved: 'Aprobado', exempt: 'Exento de LMIA' },
    englishLevel: 'Nivel de Ingles',
    frenchLevel: 'Nivel de Frances',
    langLevels: { none: 'Ninguno', basic: 'Basico', intermediate: 'Intermedio', advanced: 'Avanzado', native: 'Nativo' },
    ieltsScore: 'Puntaje IELTS / CELPIP (si tiene)',
    tefScore: 'Puntaje TEF / TCF (si tiene)',
    familyInCanada: 'Tiene familia en Canada?',
    familyDetails: 'Quien? (nombre, relacion, su estatus)',
    criminalHistory: 'Antecedentes penales?',
    criminalDetails: 'Si es si, proporcione detalles',
    medicalIssues: 'Problemas medicos que puedan afectar su solicitud?',
    previousApps: 'Solicitudes de inmigracion anteriores',
    previousAppsDetails: 'Liste solicitudes anteriores (tipo, fecha, resultado)',
    budget: 'Presupuesto para servicios de inmigracion (CAD)',
    additionalNotes: 'Notas o Preguntas Adicionales',
    reviewTitle: 'Revise Su Informacion',
    reviewSubtitle: 'Por favor revise antes de enviar',
    downloadPdf: 'Descargar PDF',
    saved: 'Admision de cliente guardada exitosamente!',
    required: 'Requerido'
  }
}

const API_URL = import.meta.env.VITE_API_URL ||
  (window.location.hostname.includes('railway.app')
    ? 'https://immigration-case-predictor-production.up.railway.app'
    : 'http://localhost:8000')

export default function ClientIntake({ user }) {
  const [lang, setLang] = useState('en')
  const [step, setStep] = useState(1)
  const [data, setData] = useState({})
  const [services, setServices] = useState({})
  const [children, setChildren] = useState([])
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState(false)
  const [saveStatus, setSaveStatus] = useState('')
  const l = t[lang]

  // Load saved draft on mount
  useEffect(() => {
    if (user?.id && user.id !== 'admin') {
      supabase.from('client_intakes').select('*').eq('user_id', user.id).eq('status', 'draft').order('updated_at', { ascending: false }).limit(1).then(({ data: rows }) => {
        if (rows && rows.length > 0) {
          const saved = rows[0]
          if (saved.intake_data) setData(saved.intake_data)
          if (saved.services) setServices(saved.services)
          if (saved.intake_data?.children) setChildren(saved.intake_data.children)
          setSaveStatus('Draft loaded')
          setTimeout(() => setSaveStatus(''), 3000)
        }
      })
    }
  }, [])

  // Auto-save when data changes
  const saveTimerRef = useRef(null)
  const isSavingRef = useRef(false)

  const autoSave = async () => {
    if (!user?.id || user.id === 'admin' || isSavingRef.current) return
    isSavingRef.current = true
    try {
      const { data: existing } = await supabase.from('client_intakes').select('id').eq('user_id', user.id).eq('status', 'draft').limit(1)
      const payload = { user_id: user.id, intake_data: { ...data, children }, services, status: 'draft', updated_at: new Date().toISOString() }
      if (existing && existing.length > 0) {
        await supabase.from('client_intakes').update(payload).eq('id', existing[0].id)
      } else {
        await supabase.from('client_intakes').insert(payload)
      }
      setSaveStatus('Auto-saved')
      setTimeout(() => setSaveStatus(''), 2000)
    } catch (e) { console.error('Auto-save error:', e) }
    isSavingRef.current = false
  }

  const triggerSave = () => {
    if (saveTimerRef.current) clearTimeout(saveTimerRef.current)
    saveTimerRef.current = setTimeout(autoSave, 3000)
  }

  const u = (field, value) => {
    setData(prev => ({ ...prev, [field]: value }))
    triggerSave()
  }
  const toggleService = (key) => {
    setServices(prev => ({ ...prev, [key]: !prev[key] }))
    triggerSave()
  }

  // Children/Dependents helpers
  const addChild = () => {
    setChildren(prev => [...prev, { name: '', dob: '', gender: '', relationship: '', citizenship: '', passport_number: '', passport_expiry: '', country_birth: '', marital_status: '', accompanying: 'yes' }])
    triggerSave()
  }
  const updateChild = (index, field, value) => {
    setChildren(prev => prev.map((child, i) => i === index ? { ...child, [field]: value } : child))
    triggerSave()
  }
  const removeChild = (index) => {
    setChildren(prev => prev.filter((_, i) => i !== index))
    triggerSave()
  }

  const steps = [l.step1, l.step2, l.step3, l.step4, l.step5, l.step6, l.step7]
  const totalSteps = steps.length

  const renderField = (label, field, type = 'text', required = false, placeholder = '', options = null) => (
    <div className="mb-4" key={field}>
      <label className="block text-sm font-medium text-slate-700 mb-1.5">{label} {required && <span className="text-red-500">*</span>}</label>
      {options ? (
        <select value={data[field] || ''} onChange={(e) => u(field, e.target.value)} className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent">
          <option value="">Select...</option>
          {options.map(opt => <option key={opt.value} value={opt.value}>{opt.label}</option>)}
        </select>
      ) : type === 'textarea' ? (
        <textarea value={data[field] || ''} onChange={(e) => u(field, e.target.value)} rows={3} className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent resize-none" placeholder={placeholder} />
      ) : (
        <input type={type} value={data[field] || ''} onChange={(e) => u(field, e.target.value)} className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent" placeholder={placeholder} />
      )}
    </div>
  )

  const handleSubmit = async () => {
    setLoading(true)
    try {
      const intakeData = { ...data, children, services_needed: Object.keys(services).filter(k => services[k]), language: lang }
      // Save to Supabase
      if (user?.id !== 'admin') {
        await supabase.from('client_intakes').insert({ user_id: user.id, intake_data: intakeData, status: 'new' })
      }
      // Download PDF
      const response = await fetch(`${API_URL}/generate-intake-pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(intakeData)
      })
      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `client_intake_${data.full_name || 'form'}.pdf`
        a.click()
        window.URL.revokeObjectURL(url)
      }
      setSuccess(true)
      setTimeout(() => setSuccess(false), 5000)
    } catch (err) {
      console.error(err)
      alert('Error saving intake form')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header with language toggle */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl p-6">
        <div className="flex justify-between items-start">
          <div>
            <h2 className="text-xl font-semibold text-slate-800 mb-2">{l.title}</h2>
            <p className="text-slate-600">{l.subtitle}</p>
          </div>
          <div className="flex gap-2">
            <button onClick={() => setLang('en')} className={`px-3 py-1.5 rounded-lg text-sm font-medium ${lang === 'en' ? 'bg-blue-600 text-white' : 'bg-white border border-slate-300 text-slate-600'}`}>English</button>
            <button onClick={() => setLang('es')} className={`px-3 py-1.5 rounded-lg text-sm font-medium ${lang === 'es' ? 'bg-blue-600 text-white' : 'bg-white border border-slate-300 text-slate-600'}`}>Español</button>
          </div>
        </div>
      </div>

      {/* Progress */}
      <div className="bg-white rounded-xl border border-slate-200 p-4">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-slate-700">{steps[step - 1]}</span>
          <span className="text-sm text-slate-500">{step} / {totalSteps}</span>
        </div>
        <div className="w-full bg-slate-200 rounded-full h-2">
          <div className="bg-red-600 h-2 rounded-full transition-all" style={{ width: `${(step / totalSteps) * 100}%` }} />
        </div>
      </div>

      {success && <div className="p-4 bg-green-50 border border-green-200 text-green-700 rounded-lg">{l.saved}</div>}
      {saveStatus && <div className="text-xs text-slate-400 text-right">{saveStatus}</div>}

      {/* Form Steps */}
      <div className="bg-white rounded-xl border border-slate-200 p-6">
        {step === 1 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6">
            {renderField(l.fullName, 'full_name', 'text', true, 'John Michael Smith')}
            {renderField(l.dob, 'dob', 'date', true)}
            {renderField(l.countryBirth, 'country_birth', 'text', true)}
            {renderField(l.citizenship, 'citizenship', 'text', true)}
            <div className="md:col-span-2">{renderField(l.currentAddress, 'address', 'text', false, '123 Main St, Toronto, ON')}</div>
            {renderField(l.phone, 'phone', 'text', false, '+1 (416) 555-1234')}
            {renderField(l.email, 'email', 'email', false, 'client@email.com')}
            {renderField(l.maritalStatus, 'marital_status', 'text', false, '', [
              { value: 'single', label: l.single },
              { value: 'married', label: l.married },
              { value: 'common_law', label: l.commonLaw },
              { value: 'divorced', label: l.divorced },
              { value: 'separated', label: l.separated },
              { value: 'widowed', label: l.widowed }
            ])}
            {renderField(l.numDependents, 'dependents', 'number', false, '0')}
          </div>
        )}

        {step === 2 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6">
            <div className="md:col-span-2">
              {renderField(l.currentStatus, 'current_status', 'text', true, '', Object.entries(l.statusOptions).map(([value, label]) => ({ value, label })))}
            </div>
            {renderField(l.permitNumber, 'permit_number', 'text', false, 'T123456789')}
            {renderField(l.dateIssued, 'date_issued', 'date')}
            {renderField(l.expiryDate, 'expiry_date', 'date')}
            {renderField(l.howLongInCanada, 'time_in_canada', 'text', false, 'e.g., 2 years, 6 months')}
            <div className="md:col-span-2">
              {renderField(l.everRefused, 'ever_refused', 'text', false, '', [{ value: 'yes', label: l.yes }, { value: 'no', label: l.no }])}
            </div>
            {data.ever_refused === 'yes' && (
              <div className="md:col-span-2">
                {renderField(l.refusalDetails, 'refusal_details', 'textarea', false, 'Country, date, reason...')}
              </div>
            )}
          </div>
        )}

        {step === 3 && (
          <div>
            <label className="block text-sm font-medium text-slate-700 mb-3">{l.whatLookingToDo}</label>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-6">
              {Object.entries(l.serviceOptions).map(([key, label]) => (
                <button key={key} onClick={() => toggleService(key)} className={`p-3 rounded-lg border-2 text-left text-sm font-medium transition-all ${services[key] ? 'border-red-500 bg-red-50 text-red-700' : 'border-slate-200 text-slate-600 hover:border-slate-300'}`}>
                  {services[key] ? '✓ ' : ''}{label}
                </button>
              ))}
            </div>
            {services.other && renderField(l.otherDetails, 'other_service_details', 'textarea')}
          </div>
        )}

        {step === 4 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6">
            {renderField(l.education, 'education_level', 'text', false, '', Object.entries(l.eduOptions).map(([value, label]) => ({ value, label })))}
            {renderField(l.fieldOfStudy, 'field_of_study', 'text', false, 'e.g., Computer Science, Nursing')}
            {renderField(l.eduCountry, 'edu_country')}
            {renderField(l.wcaAssessed, 'eca_assessed', 'text', false, '', [{ value: 'yes', label: l.yes }, { value: 'no', label: l.no }])}
            {renderField(l.currentOccupation, 'occupation', 'text', false, 'e.g., Software Developer, Nurse')}
            {renderField(l.nocCode, 'noc_code', 'text', false, 'e.g., 21232')}
            {renderField(l.yearsExpCanada, 'years_exp_canada', 'number', false, '0')}
            {renderField(l.yearsExpTotal, 'years_exp_total', 'number', false, '0')}
            {renderField(l.hasJobOffer, 'has_job_offer', 'text', false, '', [{ value: 'yes', label: l.yes }, { value: 'no', label: l.no }])}
            {renderField(l.lmiaStatus, 'lmia_status', 'text', false, '', Object.entries(l.lmiaOptions).map(([value, label]) => ({ value, label })))}
          </div>
        )}

        {step === 5 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6">
            {renderField(l.englishLevel, 'english_level', 'text', false, '', Object.entries(l.langLevels).map(([value, label]) => ({ value, label })))}
            {renderField(l.frenchLevel, 'french_level', 'text', false, '', Object.entries(l.langLevels).map(([value, label]) => ({ value, label })))}
            {renderField(l.ieltsScore, 'ielts_score', 'text', false, 'e.g., L8.5 R7.5 W7.0 S7.5')}
            {renderField(l.tefScore, 'tef_score', 'text', false, 'e.g., 400/450')}
            {renderField(l.familyInCanada, 'family_in_canada', 'text', false, '', [{ value: 'yes', label: l.yes }, { value: 'no', label: l.no }])}
            {data.family_in_canada === 'yes' && (
              <div className="md:col-span-2">{renderField(l.familyDetails, 'family_details', 'textarea')}</div>
            )}
            {renderField(l.criminalHistory, 'criminal_history', 'text', false, '', [{ value: 'yes', label: l.yes }, { value: 'no', label: l.no }])}
            {data.criminal_history === 'yes' && (
              <div className="md:col-span-2">{renderField(l.criminalDetails, 'criminal_details', 'textarea')}</div>
            )}
            {renderField(l.medicalIssues, 'medical_issues', 'text', false, '', [{ value: 'yes', label: l.yes }, { value: 'no', label: l.no }])}
            {renderField(l.previousApps, 'previous_apps', 'text', false, '', [{ value: 'yes', label: l.yes }, { value: 'no', label: l.no }])}
            {data.previous_apps === 'yes' && (
              <div className="md:col-span-2">{renderField(l.previousAppsDetails, 'previous_apps_details', 'textarea')}</div>
            )}
            {renderField(l.budget, 'budget', 'text', false, 'e.g., $3,000 - $5,000')}
            <div className="md:col-span-2">{renderField(l.additionalNotes, 'notes', 'textarea', false, 'Any other information...')}</div>
          </div>
        )}

        {step === 6 && (
          <div>
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-slate-800 mb-1">{l.dependentsTitle}</h3>
              <p className="text-sm text-slate-500">{l.dependentsSubtitle}</p>
            </div>

            {children.length === 0 && (
              <div className="text-center py-8 bg-slate-50 rounded-lg border-2 border-dashed border-slate-200 mb-4">
                <span className="text-4xl mb-3 block">👶</span>
                <p className="text-slate-500 mb-4">No dependents added yet</p>
              </div>
            )}

            {children.map((child, index) => (
              <div key={index} className="mb-6 p-5 bg-slate-50 rounded-xl border border-slate-200 relative">
                <div className="flex justify-between items-center mb-4">
                  <h4 className="font-medium text-slate-700">Child {index + 1}</h4>
                  <button onClick={() => removeChild(index)} className="text-red-500 hover:text-red-700 text-sm font-medium">{l.removeChild} ✕</button>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">{l.childName} <span className="text-red-500">*</span></label>
                    <input type="text" value={child.name} onChange={(e) => updateChild(index, 'name', e.target.value)} className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent" placeholder="Full legal name" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">{l.childDob} <span className="text-red-500">*</span></label>
                    <input type="date" value={child.dob} onChange={(e) => updateChild(index, 'dob', e.target.value)} className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">{l.childGender}</label>
                    <select value={child.gender} onChange={(e) => updateChild(index, 'gender', e.target.value)} className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent">
                      <option value="">Select...</option>
                      {Object.entries(l.genderOptions).map(([val, label]) => <option key={val} value={val}>{label}</option>)}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">{l.childRelationship} <span className="text-red-500">*</span></label>
                    <select value={child.relationship} onChange={(e) => updateChild(index, 'relationship', e.target.value)} className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent">
                      <option value="">Select...</option>
                      {Object.entries(l.relationshipOptions).map(([val, label]) => <option key={val} value={val}>{label}</option>)}
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">{l.childCountryBirth}</label>
                    <input type="text" value={child.country_birth} onChange={(e) => updateChild(index, 'country_birth', e.target.value)} className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent" placeholder="Country" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">{l.childCitizenship}</label>
                    <input type="text" value={child.citizenship} onChange={(e) => updateChild(index, 'citizenship', e.target.value)} className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent" placeholder="Country of citizenship" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">{l.childPassportNum}</label>
                    <input type="text" value={child.passport_number} onChange={(e) => updateChild(index, 'passport_number', e.target.value)} className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent" placeholder="Passport number" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">{l.childPassportExpiry}</label>
                    <input type="date" value={child.passport_expiry} onChange={(e) => updateChild(index, 'passport_expiry', e.target.value)} className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent" />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">{l.childMaritalStatus}</label>
                    <select value={child.marital_status} onChange={(e) => updateChild(index, 'marital_status', e.target.value)} className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent">
                      <option value="">Select...</option>
                      <option value="single">{l.single}</option>
                      <option value="married">{l.married}</option>
                      <option value="common_law">{l.commonLaw}</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">{l.childAccompanying}</label>
                    <select value={child.accompanying} onChange={(e) => updateChild(index, 'accompanying', e.target.value)} className="w-full border border-slate-300 rounded-lg px-4 py-2.5 focus:ring-2 focus:ring-red-500 focus:border-transparent">
                      <option value="yes">{l.yes}</option>
                      <option value="no">{l.no}</option>
                    </select>
                  </div>
                </div>
              </div>
            ))}

            <button onClick={addChild} className="w-full py-3 border-2 border-dashed border-slate-300 rounded-xl text-slate-600 hover:border-red-400 hover:text-red-600 font-medium transition-colors">
              + {l.addChild}
            </button>
          </div>
        )}

        {step === 7 && (
          <div>
            <h3 className="text-lg font-semibold text-slate-800 mb-4">{l.reviewTitle}</h3>
            <p className="text-slate-500 mb-6">{l.reviewSubtitle}</p>
            <div className="space-y-4 text-sm">
              {Object.entries(data).filter(([_, v]) => v).map(([key, value]) => (
                <div key={key} className="flex justify-between py-2 border-b border-slate-100">
                  <span className="text-slate-500 capitalize">{key.replace(/_/g, ' ')}</span>
                  <span className="font-medium text-slate-800">{value}</span>
                </div>
              ))}
              {Object.keys(services).filter(k => services[k]).length > 0 && (
                <div className="py-2 border-b border-slate-100">
                  <span className="text-slate-500">{l.whatLookingToDo}</span>
                  <div className="mt-1 flex flex-wrap gap-2">
                    {Object.keys(services).filter(k => services[k]).map(k => (
                      <span key={k} className="px-2 py-1 bg-red-100 text-red-700 rounded text-xs">{l.serviceOptions[k]}</span>
                    ))}
                  </div>
                </div>
              )}
              {children.length > 0 && (
                <div className="py-4 border-b border-slate-100">
                  <span className="text-slate-500 font-medium">{l.dependentsTitle} ({children.length})</span>
                  <div className="mt-2 space-y-2">
                    {children.map((child, i) => (
                      <div key={i} className="bg-slate-50 rounded-lg p-3">
                        <span className="font-medium text-slate-800">{child.name || `Child ${i + 1}`}</span>
                        <span className="text-slate-400 ml-2">
                          {child.dob && `DOB: ${child.dob}`}
                          {child.relationship && ` • ${l.relationshipOptions[child.relationship] || child.relationship}`}
                          {child.citizenship && ` • ${child.citizenship}`}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Navigation */}
        <div className="flex justify-between mt-8 pt-6 border-t border-slate-200">
          <button onClick={() => setStep(s => Math.max(1, s - 1))} disabled={step === 1} className="px-6 py-2.5 border border-slate-300 rounded-lg text-slate-600 hover:bg-slate-50 disabled:opacity-50 disabled:cursor-not-allowed">{l.prev}</button>
          {step < totalSteps ? (
            <button onClick={() => setStep(s => s + 1)} className="px-6 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium">{l.next}</button>
          ) : (
            <button onClick={handleSubmit} disabled={loading} className="px-6 py-2.5 bg-green-600 hover:bg-green-700 text-white rounded-lg font-medium disabled:bg-slate-300">{loading ? l.saving : l.submit}</button>
          )}
        </div>
      </div>
    </div>
  )
}
