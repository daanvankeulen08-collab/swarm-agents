<!-- section: Client Database -->

## Client Database

The Client Database stores operational and assignment information for hospitals, clinics, and agencies. It must not contain patient names, medical record numbers, diagnosis details, or other PHI. All examples below are fictional.

### A. Client and Facility Directory

| Client ID | Organization | Facility Type | Department/Service | Primary Contact | Phone | Email | Scheduling Method | Operating Hours | Site Time Zone | Relationship Status | Security/Access Review |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CL-001 | Riverbend Regional Hospital | Hospital | Radiology | Dana Brooks, Scheduling | (555) 010-0142 | dbrooks@example.invalid | Central dispatch portal | Mon–Fri, 07:00–19:00 | Eastern | Active | Completed 2025-01-08 |
| CL-002 | Northgate Imaging Center | Imaging center | MRI/CT | Luis Ortega, Coordinator | (555) 010-0186 | lortega@example.invalid | Email and phone | Mon–Sat, 06:30–20:00 | Central | Active | Completed 2025-01-10 |
| CL-003 | Lakeside Specialty Clinic | Clinic | Cardiology | Priya Shah, Office Manager | (555) 010-0129 | pshah@example.invalid | Online request form | Mon–Fri, 08:00–17:00 | Pacific | Conditional | Follow-up required |
| CL-004 | Harborview Medical Group | Medical group | Neurology | Michael Turner, Referral Lead | (555) 010-0171 | mturner@example.invalid | Agency referral system | Mon–Fri, 07:30–18:00 | Eastern | Active | Completed 2025-01-06 |

### B. Assignment Records

| Assignment ID | Client ID | Site/Department | Assignment Type | Typical Coverage | Required Skills/Credentials | Language(s) | Scheduling Frequency | Onboarding Status | Cancellation Notice | Current Priority |
|---|---|---|---|---|---|---|---|---|---|---|
| AS-101 | CL-001 | Radiology | On-site interpreting | Mon–Fri, 08:00–16:00 | Radiology terminology; BLS/CMI or equivalent | Spanish | Weekly | Complete | 24 hours | Standard |
| AS-102 | CL-002 | MRI/CT | On-site interpreting | Mon, Wed, Fri, 07:00–15:00 | MRI/CT workflow; BLS/CMI or equivalent | Spanish | Twice weekly | Complete | 48 hours | High |
| AS-103 | CL-003 | Cardiology | Remote and on-site | Tue–Thu, 09:00–17:00 | Cardiology terminology; CPI or equivalent | Mandarin | As needed | Documentation pending | 24 hours | Standard |
| AS-104 | CL-004 | Neurology | Remote interpreting | Mon–Fri, 08:00–18:00 | Neurology terminology; BLS/CMI or equivalent | ASL | Daily | Complete | Two hours | High |

### C. Client Requirements and Security Profile

| Client ID | Documents Required Before Scheduling | Badge/ID Requirements | Secure Communication Method | Remote-Access Policy | Data Handling Rule | Review Date | Reviewer |
|---|---|---|---|---|---|---|---|
| CL-001 | Contract, liability coverage, orientation acknowledgment | Government photo ID and current client badge | Client dispatch portal | Remote sessions through approved platform only | Do not retain appointment or patient details locally | 2025-06-08 | Interpreter |
| CL-002 | Contract, liability coverage, HIPAA acknowledgment | Government photo ID; temporary vendor pass | Encrypted email for scheduling only | Home workspace must be private and secure | Remove interpreter credentials after each session | 2025-06-10 | Interpreter |
| CL-003 | Contract, liability coverage, background-check confirmation | Government photo ID | Client request form | Remote work permitted after approval | No clinical documents stored on personal devices | Pending | Interpreter |
| CL-004 | Contract, liability coverage, competency documentation | Government photo ID and current client badge | Agency referral system | Use agency platform exclusively | Lock or close notes before leaving workspace | 2025-06-06 | Interpreter |

### D. Database Maintenance Checklist

- [ ] Confirm the organization name, department, contact, phone, email, and scheduling method.
- [ ] Record operating hours, time zone, notice period, and preferred communication channel.
- [ ] Link every active assignment to a valid client ID.
- [ ] List required credentials, competencies, languages, and onboarding documents.
- [ ] Verify contract, insurance, identification, and site-access requirements.
- [ ] Record the approved communication and remote-interpreting platforms.
- [ ] Confirm that no patient identifiers or clinical records are stored.
- [ ] Review security instructions at activation and at least every six months.
- [ ] Update contact details after any staff, scheduling, or policy change.
- [ ] Archive inactive clients while retaining required business records securely.
- [ ] Back up the database using encryption and role-based access.
- [ ] Restrict access to authorized users and review access at least quarterly.

**Conclusion:** This Client Database provides complete, actionable records for facilities, contacts, assignments, requirements, and security controls. It supports dependable scheduling and compliance while keeping patient information outside the planner.
