import { createContext, useContext, useState, useEffect } from 'react'

const LangContext = createContext()

export function LangProvider({ children }) {
  const [lang, setLang] = useState(() => localStorage.getItem('app_lang') || 'en')

  useEffect(() => {
    localStorage.setItem('app_lang', lang)
  }, [lang])

  return (
    <LangContext.Provider value={{ lang, setLang }}>
      {children}
    </LangContext.Provider>
  )
}

export function useLang() {
  return useContext(LangContext)
}

// Translation helper - returns the string for current language
export function useT() {
  const { lang } = useLang()
  return (key) => {
    const entry = translations[key]
    if (!entry) return key
    return entry[lang] || entry.en || key
  }
}

// All application translations
export const translations = {
  // ===== Global / Header =====
  'app.title': { en: 'ImmigrationAI', es: 'ImmigrationAI' },
  'app.subtitle': { en: 'Legal Intelligence Platform', es: 'Plataforma de Inteligencia Legal' },
  'app.signOut': { en: 'Sign Out', es: 'Cerrar Sesión' },
  'app.loading': { en: 'Loading...', es: 'Cargando...' },

  // ===== Language Switcher =====
  'lang.en': { en: 'English', es: 'English' },
  'lang.es': { en: 'Español', es: 'Español' },

  // ===== Navigation Tabs =====
  'nav.overview': { en: 'Overview', es: 'Inicio' },
  'nav.eligibility': { en: 'Eligibility Check', es: 'Verificar Elegibilidad' },
  'nav.visaforms': { en: 'Visa Forms', es: 'Formularios de Visa' },
  'nav.sponsorship': { en: 'Sponsorship Forms', es: 'Formularios de Patrocinio' },
  'nav.predictor': { en: 'Case Predictor (Beta)', es: 'Predictor (Beta)' },
  'nav.history': { en: 'My Cases', es: 'Mis Casos' },

  // ===== Home Page =====
  'home.hero.title': { en: 'AI-Powered Immigration Case Analysis', es: 'Análisis de Casos de Inmigración con IA' },
  'home.hero.subtitle': { en: 'Leverage machine learning trained on thousands of Federal Court decisions to gain insights into case outcomes and streamline your practice.', es: 'Aproveche el aprendizaje automático entrenado con miles de decisiones del Tribunal Federal para obtener información sobre resultados de casos y optimizar su práctica.' },
  'home.hero.analyze': { en: 'Analyze a Case', es: 'Analizar un Caso' },
  'home.hero.sponsorship': { en: 'Sponsorship Forms', es: 'Formularios de Patrocinio' },
  'home.predictor.title': { en: 'Case Outcome Predictor', es: 'Predictor de Resultados de Casos' },
  'home.predictor.desc': { en: 'Our AI model analyzes case facts and predicts likely outcomes based on patterns from 7,093 Federal Court judicial review decisions.', es: 'Nuestro modelo de IA analiza los hechos del caso y predice resultados probables basándose en patrones de 7,093 decisiones de revisión judicial del Tribunal Federal.' },
  'home.predictor.f1': { en: 'Trained on real Federal Court decisions', es: 'Entrenado con decisiones reales del Tribunal Federal' },
  'home.predictor.f2': { en: 'Identifies key legal factors', es: 'Identifica factores legales clave' },
  'home.predictor.f3': { en: 'Provides confidence scoring', es: 'Proporciona puntuación de confianza' },
  'home.predictor.f4': { en: 'Historical context comparison', es: 'Comparación de contexto histórico' },
  'home.sponsorship.title': { en: 'Sponsorship Form Assistant', es: 'Asistente de Formularios de Patrocinio' },
  'home.sponsorship.desc': { en: 'Streamline spousal sponsorship applications with our guided form wizard that helps you complete IMM 1344, IMM 0008, and IMM 5532.', es: 'Simplifique las solicitudes de patrocinio conyugal con nuestro asistente guiado que le ayuda a completar IMM 1344, IMM 0008 e IMM 5532.' },
  'home.sponsorship.f1': { en: 'Step-by-step guided process', es: 'Proceso guiado paso a paso' },
  'home.sponsorship.f2': { en: 'IRCC-compliant field formats', es: 'Formatos de campo compatibles con IRCC' },
  'home.sponsorship.f3': { en: 'PDF summary generation', es: 'Generación de resumen en PDF' },
  'home.sponsorship.f4': { en: 'Save and resume applications', es: 'Guardar y continuar solicitudes' },
  'home.model.title': { en: 'About Our AI Model', es: 'Acerca de Nuestro Modelo de IA' },
  'home.model.training': { en: 'Training Data', es: 'Datos de Entrenamiento' },
  'home.model.trainingDesc': { en: 'Our model was trained on the Refugee Law Lab dataset, containing 7,093 Federal Court of Canada judicial review decisions spanning from 1996 to 2022. This comprehensive dataset includes both allowed and dismissed cases across various claim types.', es: 'Nuestro modelo fue entrenado con el conjunto de datos del Refugee Law Lab, que contiene 7,093 decisiones de revisión judicial del Tribunal Federal de Canadá desde 1996 hasta 2022. Este conjunto de datos incluye casos permitidos y rechazados de varios tipos.' },
  'home.model.totalCases': { en: 'Total Cases', es: 'Total de Casos' },
  'home.model.timePeriod': { en: 'Time Period', es: 'Período' },
  'home.model.allowedRate': { en: 'Allowed Rate', es: 'Tasa de Aprobación' },
  'home.model.dismissedRate': { en: 'Dismissed Rate', es: 'Tasa de Rechazo' },
  'home.model.architecture': { en: 'Model Architecture', es: 'Arquitectura del Modelo' },
  'home.model.archDesc': { en: 'We use DistilBERT, a state-of-the-art transformer model optimized for text classification. The model was fine-tuned specifically on immigration case law to understand legal language and identify patterns that correlate with case outcomes.', es: 'Usamos DistilBERT, un modelo transformer de última generación optimizado para clasificación de texto. El modelo fue ajustado específicamente con jurisprudencia de inmigración para entender el lenguaje legal e identificar patrones que se correlacionan con los resultados de los casos.' },
  'home.model.step1': { en: 'Text tokenization and encoding', es: 'Tokenización y codificación de texto' },
  'home.model.step2': { en: 'Transformer attention analysis', es: 'Análisis de atención del transformer' },
  'home.model.step3': { en: 'Binary classification (Allowed/Dismissed)', es: 'Clasificación binaria (Aprobado/Rechazado)' },
  'home.model.step4': { en: 'Confidence scoring and factor extraction', es: 'Puntuación de confianza y extracción de factores' },
  'home.limitations.title': { en: 'Important Limitations', es: 'Limitaciones Importantes' },
  'home.limitations.1': { en: 'This tool is designed to assist legal professionals, not replace legal judgment.', es: 'Esta herramienta está diseñada para asistir a profesionales legales, no para reemplazar el juicio legal.' },
  'home.limitations.2': { en: 'Predictions are based on historical patterns and may not account for recent legal developments.', es: 'Las predicciones se basan en patrones históricos y pueden no considerar desarrollos legales recientes.' },
  'home.limitations.3': { en: 'Each case has unique circumstances that may not be fully captured by text analysis.', es: 'Cada caso tiene circunstancias únicas que pueden no ser completamente capturadas por el análisis de texto.' },
  'home.limitations.4': { en: 'Always conduct independent legal research and analysis for your clients.', es: 'Siempre realice investigación y análisis legal independiente para sus clientes.' },

  // ===== Footer =====
  'footer.title': { en: 'ImmigrationAI', es: 'ImmigrationAI' },
  'footer.desc': { en: 'AI-powered legal intelligence for Canadian immigration professionals.', es: 'Inteligencia legal impulsada por IA para profesionales de inmigración canadiense.' },
  'footer.data': { en: 'Data Sources', es: 'Fuentes de Datos' },
  'footer.dataDesc': { en: 'Trained on 7,093 Federal Court decisions from the Refugee Law Lab dataset (1996-2022).', es: 'Entrenado con 7,093 decisiones del Tribunal Federal del conjunto de datos del Refugee Law Lab (1996-2022).' },
  'footer.disclaimer': { en: 'Disclaimer', es: 'Aviso Legal' },
  'footer.disclaimerDesc': { en: 'This tool provides informational analysis only and does not constitute legal advice.', es: 'Esta herramienta proporciona solo análisis informativo y no constituye asesoría legal.' },
  'footer.copyright': { en: '© 2026 ImmigrationAI. For professional use only.', es: '© 2026 ImmigrationAI. Solo para uso profesional.' },

  // ===== Auth Page =====
  'auth.title': { en: 'Welcome to ImmigrationAI', es: 'Bienvenido a ImmigrationAI' },
  'auth.subtitle': { en: 'Sign in to access your immigration case tools', es: 'Inicie sesión para acceder a sus herramientas de casos de inmigración' },
  'auth.email': { en: 'Email', es: 'Correo Electrónico' },
  'auth.password': { en: 'Password', es: 'Contraseña' },
  'auth.signIn': { en: 'Sign In', es: 'Iniciar Sesión' },
  'auth.signUp': { en: 'Sign Up', es: 'Registrarse' },
  'auth.noAccount': { en: "Don't have an account?", es: '¿No tiene una cuenta?' },
  'auth.hasAccount': { en: 'Already have an account?', es: '¿Ya tiene una cuenta?' },
  'auth.orContinue': { en: 'Or continue with', es: 'O continuar con' },
  'auth.google': { en: 'Google', es: 'Google' },
  'auth.adminAccess': { en: 'Admin Access (Demo)', es: 'Acceso Admin (Demo)' },
  'auth.forgotPassword': { en: 'Forgot password?', es: '¿Olvidó su contraseña?' },

  // ===== Eligibility Check =====
  'eligibility.title': { en: 'Eligibility Pre-Assessment', es: 'Pre-Evaluación de Elegibilidad' },
  'eligibility.subtitle': { en: 'Answer a few questions to check if you meet the basic requirements for your Canadian immigration application.', es: 'Responda algunas preguntas para verificar si cumple con los requisitos básicos para su solicitud de inmigración canadiense.' },
  'eligibility.visitor': { en: 'Visitor Visa', es: 'Visa de Visitante' },
  'eligibility.visitorDesc': { en: 'Tourism, visiting family, or business', es: 'Turismo, visitar familia o negocios' },
  'eligibility.work': { en: 'Work Permit', es: 'Permiso de Trabajo' },
  'eligibility.workDesc': { en: 'Employment in Canada', es: 'Empleo en Canadá' },
  'eligibility.super': { en: 'Super Visa', es: 'Super Visa' },
  'eligibility.superDesc': { en: 'Parents & grandparents (up to 5 years)', es: 'Padres y abuelos (hasta 5 años)' },
  'eligibility.next': { en: 'Next', es: 'Siguiente' },
  'eligibility.prev': { en: 'Previous', es: 'Anterior' },
  'eligibility.submit': { en: 'Get Assessment', es: 'Obtener Evaluación' },
  'eligibility.restart': { en: 'Start Over', es: 'Comenzar de Nuevo' },
  'eligibility.question': { en: 'Question', es: 'Pregunta' },
  'eligibility.of': { en: 'of', es: 'de' },

  // ===== Case Predictor =====
  'predictor.title': { en: 'Case Outcome Predictor', es: 'Predictor de Resultados de Casos' },
  'predictor.subtitle': { en: 'Enter case details to predict the likely outcome based on historical Federal Court decisions.', es: 'Ingrese los detalles del caso para predecir el resultado probable basado en decisiones históricas del Tribunal Federal.' },
  'predictor.placeholder': { en: 'Paste or type the case facts, decision summary, or key arguments here...', es: 'Pegue o escriba los hechos del caso, resumen de la decisión o argumentos clave aquí...' },
  'predictor.analyze': { en: 'Analyze Case', es: 'Analizar Caso' },
  'predictor.analyzing': { en: 'Analyzing...', es: 'Analizando...' },
  'predictor.prediction': { en: 'Prediction', es: 'Predicción' },
  'predictor.confidence': { en: 'Confidence', es: 'Confianza' },
  'predictor.factors': { en: 'Key Factors Identified', es: 'Factores Clave Identificados' },
  'predictor.allowed': { en: 'Allowed', es: 'Aprobado' },
  'predictor.dismissed': { en: 'Dismissed', es: 'Rechazado' },
  'predictor.riskLevel': { en: 'Risk Level', es: 'Nivel de Riesgo' },
  'predictor.history': { en: 'Historical Context', es: 'Contexto Histórico' },
  'predictor.newCase': { en: 'New Analysis', es: 'Nuevo Análisis' },
  'predictor.save': { en: 'Save to My Cases', es: 'Guardar en Mis Casos' },
  'predictor.saved': { en: 'Saved!', es: '¡Guardado!' },

  // ===== Sponsorship Assistant =====
  'sponsor.chat': { en: '💬 Chat Assistant', es: '💬 Asistente de Chat' },
  'sponsor.wizard': { en: '📝 Form Wizard', es: '📝 Asistente de Formularios' },
  'sponsor.checklist': { en: '✅ Document Checklist', es: '✅ Lista de Documentos' },
  'sponsor.proof': { en: '💌 Communication Evidence', es: '💌 Evidencia de Comunicación' },
  'sponsor.photos': { en: '📷 Photo Album', es: '📷 Álbum de Fotos' },
  'sponsor.copy': { en: '📋 Copy for IRCC', es: '📋 Copiar para IRCC' },
  'sponsor.reports': { en: 'Form Reports', es: 'Reportes de Formularios' },
  'sponsor.formpackage': { en: '📦 Form Package', es: '📦 Paquete de Formularios' },
  'sponsor.formguide': { en: '🇪🇸 Guía de Formularios', es: '🇪🇸 Guía de Formularios' },
  'sponsor.chatTitle': { en: 'Sponsorship Assistant', es: 'Asistente de Patrocinio' },
  'sponsor.chatSubtitle': { en: 'Ask questions about spousal sponsorship', es: 'Haga preguntas sobre patrocinio conyugal' },
  'sponsor.chatPlaceholder': { en: 'Ask about sponsorship requirements, documents, timelines...', es: 'Pregunte sobre requisitos de patrocinio, documentos, plazos...' },
  'sponsor.send': { en: 'Send', es: 'Enviar' },
  'sponsor.thinking': { en: 'Thinking...', es: 'Pensando...' },
  'sponsor.examples': { en: 'Example Inputs', es: 'Ejemplos de Entrada' },
  'sponsor.irccFormat': { en: 'IRCC Format', es: 'Formato IRCC' },
  'sponsor.progress': { en: 'Progress', es: 'Progreso' },
  'sponsor.sponsorInfo': { en: 'Sponsor Info', es: 'Info del Patrocinador' },
  'sponsor.applicantInfo': { en: 'Applicant Info', es: 'Info del Solicitante' },
  'sponsor.relationship': { en: 'Relationship', es: 'Relación' },
  'sponsor.sponsorTitle': { en: 'Sponsor Information', es: 'Información del Patrocinador' },
  'sponsor.applicantTitle': { en: 'Principal Applicant Information', es: 'Información del Solicitante Principal' },
  'sponsor.relationshipTitle': { en: 'Relationship Details', es: 'Detalles de la Relación' },
  'sponsor.step': { en: 'Step', es: 'Paso' },
  'sponsor.of': { en: 'of', es: 'de' },
  'sponsor.previous': { en: 'Previous', es: 'Anterior' },
  'sponsor.continue': { en: 'Continue', es: 'Continuar' },
  'sponsor.saveDraft': { en: 'Save Draft', es: 'Guardar Borrador' },
  'sponsor.saving': { en: 'Saving...', es: 'Guardando...' },
  'sponsor.downloadPdf': { en: 'Download PDF Summary', es: 'Descargar Resumen PDF' },
  'sponsor.generating': { en: 'Generating...', es: 'Generando...' },
  'sponsor.formSaved': { en: 'Form saved successfully!', es: '¡Formulario guardado exitosamente!' },

  // ===== Sponsorship Form Fields =====
  'field.familyName': { en: 'Family Name (Surname)', es: 'Apellido(s)' },
  'field.givenName': { en: 'Given Name(s)', es: 'Nombre(s) de Pila' },
  'field.dob': { en: 'Date of Birth', es: 'Fecha de Nacimiento' },
  'field.countryBirth': { en: 'Country of Birth', es: 'País de Nacimiento' },
  'field.citizenship': { en: 'Citizenship Status', es: 'Estado de Ciudadanía' },
  'field.countryCitizenship': { en: 'Country of Citizenship', es: 'País de Ciudadanía' },
  'field.address': { en: 'Current Mailing Address', es: 'Dirección Postal Actual' },
  'field.email': { en: 'Email Address', es: 'Correo Electrónico' },
  'field.phone': { en: 'Phone Number', es: 'Número de Teléfono' },
  'field.residence': { en: 'Current Country of Residence', es: 'País de Residencia Actual' },
  'field.passport': { en: 'Passport Number', es: 'Número de Pasaporte' },
  'field.relationshipType': { en: 'Relationship Type', es: 'Tipo de Relación' },
  'field.dateMarried': { en: 'Date of Marriage/Union', es: 'Fecha de Matrimonio/Unión' },
  'field.placeMarried': { en: 'Place of Marriage', es: 'Lugar de Matrimonio' },
  'field.howMet': { en: 'How did you meet?', es: '¿Cómo se conocieron?' },
  'field.relationshipHistory': { en: 'Relationship History', es: 'Historia de la Relación' },
  'field.spouse': { en: 'Spouse', es: 'Esposo/a' },
  'field.commonLaw': { en: 'Common-law Partner', es: 'Pareja de Hecho' },
  'field.conjugal': { en: 'Conjugal Partner', es: 'Pareja Conyugal' },

  // ===== Copy for IRCC =====
  'copy.title': { en: 'Copy Your Information for IRCC', es: 'Copie Su Información para IRCC' },
  'copy.subtitle': { en: 'Click the copy button next to each field to copy it to your clipboard, then paste into the IRCC portal.', es: 'Haga clic en el botón de copiar junto a cada campo para copiarlo al portapapeles, luego péguelo en el portal de IRCC.' },
  'copy.copyAll': { en: '📋 Copy All Fields', es: '📋 Copiar Todos los Campos' },
  'copy.copiedAll': { en: '✓ Copied All!', es: '✓ ¡Todo Copiado!' },
  'copy.copy': { en: '📋 Copy', es: '📋 Copiar' },
  'copy.copied': { en: '✓ Copied!', es: '✓ ¡Copiado!' },
  'copy.noData': { en: 'No Data to Copy', es: 'Sin Datos para Copiar' },
  'copy.noDataDesc': { en: 'Fill out the form using the Chat Assistant or Form Wizard first, then come back here to copy your information for the IRCC portal.', es: 'Complete el formulario usando el Asistente de Chat o el Asistente de Formularios primero, luego regrese aquí para copiar su información para el portal de IRCC.' },
  'copy.portalLinks': { en: '🔗 IRCC Portal Links', es: '🔗 Enlaces al Portal de IRCC' },
  'copy.sponsorshipOverview': { en: '→ Spousal Sponsorship Overview', es: '→ Resumen de Patrocinio Conyugal' },
  'copy.irccAccount': { en: '→ IRCC Online Account (Sign In/Create)', es: '→ Cuenta en Línea de IRCC (Iniciar/Crear)' },
  'copy.howToApply': { en: '→ How to Apply for Spousal Sponsorship', es: '→ Cómo Solicitar Patrocinio Conyugal' },

  // ===== Document Checklist =====
  'checklist.title': { en: 'Document Checklist', es: 'Lista de Documentos' },
  'checklist.subtitle': { en: 'Track all required documents for your sponsorship application', es: 'Rastree todos los documentos requeridos para su solicitud de patrocinio' },

  // ===== Photo Album =====
  'photos.title': { en: 'Relationship Photo Album', es: 'Álbum de Fotos de la Relación' },
  'photos.subtitle': { en: 'Organize 20 photographs that tell the story of your relationship.', es: 'Organice 20 fotografías que cuenten la historia de su relación.' },
  'photos.download': { en: 'Download Photo Album PDF', es: 'Descargar Álbum de Fotos PDF' },
  'photos.downloadWord': { en: 'Download as Word (.docx)', es: 'Descargar como Word (.docx)' },
  'photos.tips': { en: 'Photo Tips for IRCC', es: 'Consejos de Fotos para IRCC' },
  'photos.whatToInclude': { en: 'What to Include:', es: 'Qué Incluir:' },
  'photos.bestPractices': { en: 'Best Practices:', es: 'Mejores Prácticas:' },
  'photos.date': { en: 'Date', es: 'Fecha' },
  'photos.location': { en: 'Location', es: 'Ubicación' },
  'photos.people': { en: 'Who is in this photo?', es: '¿Quién está en esta foto?' },
  'photos.description': { en: 'Description', es: 'Descripción' },
  'photos.save': { en: 'Save', es: 'Guardar' },
  'photos.saveAs': { en: 'Save As...', es: 'Guardar Como...' },
  'photos.new': { en: 'New', es: 'Nuevo' },
  'photos.workingOn': { en: 'Working on:', es: 'Trabajando en:' },
  'photos.photosAdded': { en: 'Photos Added', es: 'Fotos Agregadas' },

  // ===== User History =====
  'history.title': { en: 'My Cases', es: 'Mis Casos' },
  'history.subtitle': { en: 'Your saved case analyses and predictions', es: 'Sus análisis y predicciones de casos guardados' },
  'history.noCases': { en: 'No saved cases yet', es: 'Aún no hay casos guardados' },
  'history.noCasesDesc': { en: 'Analyze a case using the Case Predictor and save it to see it here.', es: 'Analice un caso usando el Predictor de Casos y guárdelo para verlo aquí.' },
  'history.delete': { en: 'Delete', es: 'Eliminar' },
  'history.viewDetails': { en: 'View Details', es: 'Ver Detalles' },

  // ===== Visa Forms =====
  'visa.title': { en: 'Visa Application Forms', es: 'Formularios de Solicitud de Visa' },
  'visa.subtitle': { en: 'Complete your visa application with our guided assistant', es: 'Complete su solicitud de visa con nuestro asistente guiado' },

  // ===== Form Package =====
  'package.title': { en: 'IRCC Form Package', es: 'Paquete de Formularios IRCC' },
  'package.subtitle': { en: 'Download and track all required forms for your application type', es: 'Descargue y rastree todos los formularios requeridos para su tipo de solicitud' },
  'package.download': { en: 'Download PDF', es: 'Descargar PDF' },
  'package.markComplete': { en: 'Mark Complete', es: 'Marcar Completo' },
  'package.completed': { en: 'Completed', es: 'Completado' },
  'package.required': { en: 'Required', es: 'Requerido' },
  'package.optional': { en: 'Optional', es: 'Opcional' },

  // ===== Common =====
  'common.yes': { en: 'Yes', es: 'Sí' },
  'common.no': { en: 'No', es: 'No' },
  'common.save': { en: 'Save', es: 'Guardar' },
  'common.cancel': { en: 'Cancel', es: 'Cancelar' },
  'common.delete': { en: 'Delete', es: 'Eliminar' },
  'common.edit': { en: 'Edit', es: 'Editar' },
  'common.close': { en: 'Close', es: 'Cerrar' },
  'common.back': { en: 'Back', es: 'Volver' },
  'common.next': { en: 'Next', es: 'Siguiente' },
  'common.previous': { en: 'Previous', es: 'Anterior' },
  'common.submit': { en: 'Submit', es: 'Enviar' },
  'common.required': { en: 'Required', es: 'Requerido' },
  'common.optional': { en: 'Optional', es: 'Opcional' },
  'common.loading': { en: 'Loading...', es: 'Cargando...' },
  'common.error': { en: 'An error occurred', es: 'Ocurrió un error' },
  'common.success': { en: 'Success!', es: '¡Éxito!' },
  'common.select': { en: 'Select...', es: 'Seleccionar...' },
}
