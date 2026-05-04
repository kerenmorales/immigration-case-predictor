import { useState } from 'react'
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
    step6: 'Review & Submit',
    next: 'Next',
    prev: 'Previous',
    submit: 'Submit & Download PDF',
    saving: 'Saving...',
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
    step6: 'Revisar y Enviar',
    next: 'Siguiente',
    prev: 'Anterior',
    submit: 'Enviar y Descargar PDF',
    saving: 'Guardando...',
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
  const [loading, setLoading] = useState(false)
  const [success, setSuccess] = useState(false)
  const l = t[lang]

  const u = (field, value) => setData(prev => ({ ...prev, [field]: value }))
  const toggleService = (key) => setServices(prev => ({ ...prev, [key]: !prev[key] }))

  const steps = [l.step1, l.step2, l.step3, l.step4, l.step5, l.step6]
  const totalSteps = steps.length

  const Field = ({ label, field, type = 'text', required = false, placeholder = '', options = null }) => (
    <div className="mb-4">
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
      const intakeData = { ...data, services_needed: Object.keys(services).filter(k => services[k]), language: lang }
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

      {/* Form Steps */}
      <div className="bg-white rounded-xl border border-slate-200 p-6">
        {step === 1 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6">
            <Field label={l.fullName} field="full_name" required placeholder="John Michael Smith" />
            <Field label={l.dob} field="dob" type="date" required />
            <Field label={l.countryBirth} field="country_birth" required />
            <Field label={l.citizenship} field="citizenship" required />
            <div className="md:col-span-2"><Field label={l.currentAddress} field="address" placeholder="123 Main St, Toronto, ON" /></div>
            <Field label={l.phone} field="phone" placeholder="+1 (416) 555-1234" />
            <Field label={l.email} field="email" type="email" placeholder="client@email.com" />
            <Field label={l.maritalStatus} field="marital_status" options={[
              { value: 'single', label: l.single },
              { value: 'married', label: l.married },
              { value: 'common_law', label: l.commonLaw },
              { value: 'divorced', label: l.divorced },
              { value: 'separated', label: l.separated },
              { value: 'widowed', label: l.widowed }
            ]} />
            <Field label={l.numDependents} field="dependents" type="number" placeholder="0" />
          </div>
        )}

        {step === 2 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6">
            <div className="md:col-span-2">
              <Field label={l.currentStatus} field="current_status" required options={Object.entries(l.statusOptions).map(([value, label]) => ({ value, label }))} />
            </div>
            <Field label={l.permitNumber} field="permit_number" placeholder="T123456789" />
            <Field label={l.dateIssued} field="date_issued" type="date" />
            <Field label={l.expiryDate} field="expiry_date" type="date" />
            <Field label={l.howLongInCanada} field="time_in_canada" placeholder="e.g., 2 years, 6 months" />
            <div className="md:col-span-2">
              <Field label={l.everRefused} field="ever_refused" options={[{ value: 'yes', label: l.yes }, { value: 'no', label: l.no }]} />
            </div>
            {data.ever_refused === 'yes' && (
              <div className="md:col-span-2">
                <Field label={l.refusalDetails} field="refusal_details" type="textarea" placeholder="Country, date, reason..." />
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
            {services.other && <Field label={l.otherDetails} field="other_service_details" type="textarea" />}
          </div>
        )}

        {step === 4 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6">
            <Field label={l.education} field="education_level" options={Object.entries(l.eduOptions).map(([value, label]) => ({ value, label }))} />
            <Field label={l.fieldOfStudy} field="field_of_study" placeholder="e.g., Computer Science, Nursing" />
            <Field label={l.eduCountry} field="edu_country" />
            <Field label={l.wcaAssessed} field="eca_assessed" options={[{ value: 'yes', label: l.yes }, { value: 'no', label: l.no }]} />
            <Field label={l.currentOccupation} field="occupation" placeholder="e.g., Software Developer, Nurse" />
            <Field label={l.nocCode} field="noc_code" placeholder="e.g., 21232" />
            <Field label={l.yearsExpCanada} field="years_exp_canada" type="number" placeholder="0" />
            <Field label={l.yearsExpTotal} field="years_exp_total" type="number" placeholder="0" />
            <Field label={l.hasJobOffer} field="has_job_offer" options={[{ value: 'yes', label: l.yes }, { value: 'no', label: l.no }]} />
            <Field label={l.lmiaStatus} field="lmia_status" options={Object.entries(l.lmiaOptions).map(([value, label]) => ({ value, label }))} />
          </div>
        )}

        {step === 5 && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6">
            <Field label={l.englishLevel} field="english_level" options={Object.entries(l.langLevels).map(([value, label]) => ({ value, label }))} />
            <Field label={l.frenchLevel} field="french_level" options={Object.entries(l.langLevels).map(([value, label]) => ({ value, label }))} />
            <Field label={l.ieltsScore} field="ielts_score" placeholder="e.g., L8.5 R7.5 W7.0 S7.5" />
            <Field label={l.tefScore} field="tef_score" placeholder="e.g., 400/450" />
            <Field label={l.familyInCanada} field="family_in_canada" options={[{ value: 'yes', label: l.yes }, { value: 'no', label: l.no }]} />
            {data.family_in_canada === 'yes' && (
              <div className="md:col-span-2"><Field label={l.familyDetails} field="family_details" type="textarea" /></div>
            )}
            <Field label={l.criminalHistory} field="criminal_history" options={[{ value: 'yes', label: l.yes }, { value: 'no', label: l.no }]} />
            {data.criminal_history === 'yes' && (
              <div className="md:col-span-2"><Field label={l.criminalDetails} field="criminal_details" type="textarea" /></div>
            )}
            <Field label={l.medicalIssues} field="medical_issues" options={[{ value: 'yes', label: l.yes }, { value: 'no', label: l.no }]} />
            <Field label={l.previousApps} field="previous_apps" options={[{ value: 'yes', label: l.yes }, { value: 'no', label: l.no }]} />
            {data.previous_apps === 'yes' && (
              <div className="md:col-span-2"><Field label={l.previousAppsDetails} field="previous_apps_details" type="textarea" /></div>
            )}
            <Field label={l.budget} field="budget" placeholder="e.g., $3,000 - $5,000" />
            <div className="md:col-span-2"><Field label={l.additionalNotes} field="notes" type="textarea" placeholder="Any other information..." /></div>
          </div>
        )}

        {step === 6 && (
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
