<!-- section: Client Database -->

## Client Database

This database stores operational and contact information for hospitals, clinics, and telehealth providers. All example records below are fictional and contain no patient information. Use “—” for fields that do not apply.

### 1. Client and Contract Directory

| Client ID | Organization / Site | Department or Service | Primary Contact | Phone | Email | Contract / Payment | Access / SLA Requirements | Status |
|---|---|---|---|---|---|---|---|---|
| CL-001 | St. Anne Medical Center — East Campus | Cardiology | Dana Ruiz | 555-014-2210 | dana.ruiz@example.org | Master services agreement; Net 30 | Report to security by 7:30 AM | Active |
| CL-002 | Riverside Imaging Center | Radiology | Omar Patel | 555-014-8834 | o.patel@example.org | Per-hour agreement; weekly invoice | Remote-session platform training required | Active |
| CL-003 | Northstar Virtual Care | Multi-specialty telemedicine | Sofia Bennett | 555-014-7621 | scheduling@example.org | Independent contractor agreement | Secure video platform; 15-minute connection test | Probation |
| — | — | — | — | — | — | — | — | — |

### 2. Assignment Schedule

| Assignment ID | Client ID | Date | Scheduled Time | Modality / Specialty | Location / Platform | Interpreter | Confirmation | Travel Minutes | Status | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| AS-101 | CL-001 | Jan 6 | 8:00 AM–12:00 PM | On-site Spanish; cardiology | Clinic B | Avery Morgan | Confirmed | 25 | Scheduled | Security desk opens at 7:30 AM |
| AS-102 | CL-002 | Jan 7 | 9:00 AM–1:00 PM | Remote Spanish; radiology | Riverside secure platform | Avery Morgan | Confirmed | 0 | Scheduled | Test audio before session |
| AS-103 | CL-003 | Jan 8 | 1:00 PM–5:00 PM | Video Spanish; follow-up visits | Northstar platform | Jordan Lee | Pending | 0 | Tentative | Client to confirm platform link |
| — | — | — | — | — | — | — | — | — | — | — |

### 3. Client Contacts and Escalation

| Contact ID | Client ID | Name | Role | Phone | Email | Preferred Channel | Backup Contact | Escalation Window |
|---|---|---|---|---|---|---|---|---|
| CT-11 | CL-001 | Dana Ruiz | Interpreter Coordinator | 555-014-2210 | dana.ruiz@example.org | Phone | Security desk: 555-014-2200 | 7:00 AM–5:00 PM |
| CT-12 | CL-002 | Omar Patel | Imaging Operations Lead | 555-014-8834 | o.patel@example.org | Email | IT support: 555-014-8851 | 8:00 AM–6:00 PM |
| CT-13 | CL-003 | Sofia Bennett | Virtual Scheduling Manager | 555-014-7621 | scheduling@example.org | Secure email | On-call coordinator: 555-014-7609 | 8:00 AM–8:00 PM |
| — | — | — | — | — | — | — | — | — |

### 4. Credentials and Client-Specific Requirements

| Record ID | Client ID | Interpreter | Requirement | Evidence / Reference | Verified Date | Expiration Date | Reviewer | Status |
|---|---|---|---|---|---|---|---|---|
| CR-21 | CL-001 | Avery Morgan | Background screening | Screening certificate #BC-1048 | Jan 2 | Jan 2 next year | Dana Ruiz | Valid |
| CR-22 | CL-002 | Avery Morgan | HIPAA training | Training record #TR-8831 | Dec 18 | Dec 18 next year | Omar Patel | Valid |
| CR-23 | CL-003 | Jordan Lee | Platform orientation | Northstar orientation receipt | Jan 4 | No expiration | Sofia Bennett | Complete |
| — | — | — | — | — | — | — | — | — |

### 5. Communication Log

| Log ID | Date / Time | Client ID | Assignment ID | Channel | Purpose | Summary and Required Action | Owner | Follow-Up Date | Status |
|---|---|---|---|---|---|---|---|---|---|
| LG-31 | Jan 3, 10:15 AM | CL-001 | AS-101 | Phone | Arrival procedure | Security desk will provide temporary badge | Avery Morgan | Jan 6 | Open |
| LG-32 | Jan 4, 2:40 PM | CL-002 | AS-102 | Email | Platform check | Audio settings passed; connection test recorded | Omar Patel | Jan 7 | Closed |
| LG-33 | Jan 5, 9:05 AM | CL-003 | AS-103 | Secure email | Confirmation | Link will be sent 30 minutes before start | Jordan Lee | Jan 8 | Open |
| — | — | — | — | — | — | — | — | — | — |

### Client Database Checklists
