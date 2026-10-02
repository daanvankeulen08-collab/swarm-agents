<!-- section: Client Database -->

## Client Database

Use one record per hospital, clinic, agency, or facility. The fictional examples below illustrate how to enter data; they are not live client information.

### 1. Client and Contact Directory

| Client ID | Organization / Site | Primary Contact | Contact Details | Assignment Type | Interpretation Needs | Preferred Communication | Status |
|---|---|---|---|---|---|---|---|
| CL-001 | Riverside Regional Hospital | Dana Ortiz, Scheduling Coordinator | 555-010-1842; scheduling@riverside-example.org | Onsite, recurring | Spanish; cardiology and discharge instructions | Email; confirm by phone | Active |
| CL-002 | North Valley Imaging Center | Priya Shah, Office Manager | 555-010-2264; pshah@northvalley-example.org | Onsite, scheduled | Mandarin; radiology and ultrasound | Text before shift | Active |
| CL-003 | Lakeside Surgical Group | Marcus Bell, Practice Administrator | 555-010-3198; mbell@lakesidesurgery-example.org | Remote and onsite | ASL; pre-op and post-op instructions | Email | Pending credentialing |
| CL-004 | Harbor Pediatrics Clinic | Elena Ruiz, Clinic Lead | 555-010-4721; eruiz@harborpediatrics-example.org | Onsite, recurring | Vietnamese; pediatric visits | Phone for changes | Active |

### 2. Weekly Availability and Scheduling Rules

| Client ID | Monday | Tuesday | Wednesday | Thursday | Friday | Minimum Notice | Cancellation Rule | Standing Blackouts |
|---|---|---|---|---|---|---|---|---|
| CL-001 | 08:00–16:00 | 08:00–16:00 | Not available | 08:00–16:00 | 08:00–12:00 | 72 hours | Client must pay 50% if canceled inside 24 hours | Department rounds start at 09:00 |
| CL-002 | Not available | 09:00–17:00 | 09:00–17:00 | 09:00–17:00 | 08:00–14:00 | 48 hours | Reschedule before 12:00 on assignment day | No Wednesday access before 10:00 |
| CL-003 | 07:00–15:00 | 07:00–15:00 | 07:00–15:00 | 07:00–15:00 | Not available | 5 business days | Approval required for changes within 48 hours | Not applicable |
| CL-004 | 08:30–15:30 | 08:30–15:30 | 08:30–15:30 | 08:30–15:30 | 08:30–15:30 | 24 hours | Same-day cancellation receives no payment | Lunch coverage required by 12:00 |

### 3. Assignments and Access Requirements

| Assignment ID | Client ID | Date and Time | Service | Location / Platform | Required Skills or Access | Travel / Parking | Confirmation Status |
|---|---|---|---|---|---|---|---|
| AS-101 | CL-001 | Mar 10, 2026, 09:00–15:00 | Cardiology rounds | Riverside, Room 4B | Spanish; HIPAA training; EHR access | Park in staff garage; allow 15 minutes | Confirmed Mar 2 |
| AS-102 | CL-002 | Mar 11, 2026, 10:00–16:00 | Ultrasound appointments | North Valley, Suite 2 | Mandarin; PACS read-only access | Onsite; parking validation available | Confirmed Mar 3 |
| AS-103 | CL-003 | Mar 12, 2026, 08:00–14:00 | Surgical consent | Secure video platform | ASL; background-check clearance | Remote; no travel | Pending badge approval |
| AS-104 | CL-004 | Mar 13, 2026, 09:00–13:00 | Pediatric well visits | Harbor, Exam 6 | Vietnamese; pediatric competency | Staff lot; use visitor entrance | Confirmed Mar 4 |

### 4. Credentials and Compliance by Client

| Client ID | Credential / Requirement | Expiration or Completion Date | Verification Status | Document Owner | Renewal Lead Time | Next Action |
|---|---|---|---|---|---|---|
| CL-001 | HIPAA attestation | Dec 31, 2026 | Verified | Interpreter | 30 days | Upload annual refresher |
| CL-002 | Background screening | Jun 30, 2027 | Verified | Interpreter | 45 days | Confirm PACS training |
| CL-003 | Facility security clearance | Not assigned; client pending review | Submitted | Client | Not specified | Follow up with Marcus |
| CL-004 | Pediatric-services competency | Aug 31, 2026 | Verified | Interpreter | 30 days | Schedule refresher |

### 5. Contact Log

| Date | Client ID | Method | Contact | Purpose | Outcome / Follow-Up |
|---|---|---|---|---|---|
| Mar 2, 2026 | CL-001 | Email | Dana Ortiz | Confirm cardiology rounds | Assignment AS-101 confirmed; parking instructions sent |
| Mar 3, 2026 | CL-002 | Phone | Priya Shah | Clarify PACS access | Read-only access approved; escort required on first shift |
| Mar 4, 2026 | CL-003 | Email | Marcus Bell | Check badge status | Security review still pending; follow up in two business days |
| Mar 5, 2026 | CL-004 | Text | Elena Ruiz | Confirm clinic coverage | Lunch interpreter assigned; no action required |

### Client Database Checklists

**New-client onboarding**
- [ ] Organization, department, address, and contact details recorded.
- [ ] Languages, specialties, and service location documented.
- [ ] Rates, cancellation terms, invoicing rules, and payment cycle saved.
- [ ] Credential, training, badge, EHR, and confidentiality requirements verified.
- [ ] Availability, notice requirements, parking, check-in, and emergency contacts saved.
- [ ] Test communication and receive written approval.

**Before every assignment**
- [ ] Confirm date, time, location, department, and expected duration.
- [ ] Review specialty, language, and required credentials.
- [ ] Check access badges, software, parking, check-in, and contact instructions.
- [ ] Send availability confirmation and identify backup arrangements.
- [ ] Add travel, preparation, and recovery time to the weekly calendar.

**Weekly review**
- [ ] Reconcile confirmed, pending, canceled, and completed assignments.
- [ ] Check upcoming credential expirations and client-specific requirements.
- [ ] Review contact records and unresolved follow-ups.
- [ ] Record hours, approved expenses, mileage, and invoice status.
- [ ] Resolve scheduling conflicts before publishing the next week.

**Conclusion:** A complete client database centralizes contacts, recurring availability, assignments, access requirements, compliance dates, and communication history. Reviewed weekly, it supports dependable scheduling, credential readiness, and accurate billing across every interpreter assignment.

<!-- REVIEW NOTE: section 'Client Database' failed completeness validation; verify tables/ending manually. -->
