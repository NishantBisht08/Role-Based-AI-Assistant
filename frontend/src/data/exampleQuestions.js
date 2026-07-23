export const exampleQuestions = {
    hr: {
        title: "HR",
        sections: [
            {
                title: "Basic Data Understanding",
                questions: [
                    "What fields are present in the employee dataset?",
                    "What is the employee ID format used in this dataset?",
                    "Which departments are present in the company?",
                    "What is the salary of Aadhya Patel?",
                    "Which city does Isha Chowdhury (FINEMP1001) work from?",
                    "What is the role of FINEMP1010?",
                    "How many leaves has FINEMP1018 taken?",
                    "What is the email format used across all employees?",
                    "Who is the manager of Sakshi Malhotra (FINEMP1002)?",
                    "What is the performance rating of FINEMP1011?"
                ]
            },
            {
                title: "Retrieval & Reasoning",
                questions: [
                    "List all employees in the Technology department.",
                    "Who has the highest salary in the dataset?",
                    "Which employees have a performance rating of 5?",
                    "How many employees are based in Delhi?",
                    "Who joined the company most recently?",
                    "Which employees have taken more leaves than their leave balance?",
                    "Who has a leave balance of 0?",
                    "List all managers (employees who appear as manager_id for others).",
                    "Which department has the most employees?",
                    "Who has the highest attendance percentage?",
                    "Find all employees who joined before 2019.",
                    "Which employees report to FINEMP1007?",
                    "Who has the lowest performance rating and highest salary?",
                    "List employees whose last review was in 2025.",
                    "Which locations have more than 5 employees?"
                ]
            }
        ]
    },

    finance: {
        title: "Finance",
        sections: [
            {
                title: "Direct Lookups",
                questions: [
                    "What was FinSolve Technologies' total revenue in 2024?",
                    "What was the net income in Q3 2024?",
                    "What was the gross margin in Q4 2024?",
                    "How much did the company spend on vendor services in Q1 2024?",
                    "What was the cash flow from operations for the full year 2024?",
                    "What was the marketing spend in Q2 2024?",
                    "What was the operating income in Q2 2024?",
                    "How much was spent on software subscriptions in Q3 2024?",
                    "What were the employee benefits and HR costs in Q4 2024?",
                    "What was the ROI for 2024?"
                ]
            },
            {
                title: "Comparison & Trend",
                questions: [
                    "Which quarter had the highest revenue in 2024?",
                    "How did gross margin change from Q1 to Q4 2024?",
                    "Which quarter had the highest net income?",
                    "How did vendor costs trend across all 4 quarters?",
                    "What was the YoY revenue growth percentage in 2024?",
                    "How did cash flow from operations change from Q1 to Q4?",
                    "Which expense category saw the highest increase in 2024 — vendor services, software subscriptions, or HR costs?",
                    "How does FinSolve's gross margin compare to the industry benchmark?",
                    "What was the biggest risk identified in Q2 and how was it mitigated?",
                    "What were the top 2 recommendations made for 2025?"
                ]
            }
        ]
    },

    employee: {
        title: "Employee",
        sections: [
            {
                title: "Direct Lookups",
                questions: [
                    "What is FinSolve Technologies' company vision?",
                    "When was FinSolve Technologies founded?",
                    "How many days of sick leave are employees entitled to per year?",
                    "What is the maternity leave entitlement for the first two children?",
                    "How many days of paternity leave does the company provide?",
                    "What percentage of CTC is the basic salary?",
                    "On which day is salary credited each month?",
                    "What is the maximum hotel reimbursement per night during official travel?",
                    "How much is the employee referral reward?",
                    "How many days of work from home are allowed per week?"
                ]
            },
            {
                title: "Policy Understanding",
                questions: [
                    "What documents are required during the pre-joining onboarding process?",
                    "How many days in advance must leave be applied for, except in emergencies?",
                    "What happens if an employee is late to work three or more times in a month?",
                    "What is the process to raise a payroll discrepancy?",
                    "How much can an employee claim for certification reimbursement per year, and what is the condition attached?",
                    "What are the steps in the conflict resolution process?",
                    "What disciplinary actions are taken for underperformance?",
                    "How is overtime compensated as per the handbook?",
                    "What is the full and final settlement timeline after exit?",
                    "What is the dress code policy from Monday to Thursday vs Friday?"
                ]
            }
        ]
    },

    engineering: {
        title: "Engineering",
        sections: [
            {
                title: "Direct Lookups",
                questions: [
                    "Where is FinSolve Technologies headquartered?",
                    "What year was FinSolve Technologies founded?",
                    "How many individual users does FinSolve serve globally?",
                    "What is the primary cloud provider used by FinSolve?",
                    "Which database is used for transactional data requiring ACID compliance?",
                    "What is the target API response time at P95?",
                    "What is the uptime target for FinSolve's systems?",
                    "What authentication protocol does FinSolve use?",
                    "What is the Recovery Time Objective (RTO) for disaster recovery?",
                    "What is the Recovery Point Objective (RPO)?"
                ]
            },
            {
                title: "Understanding & Reasoning",
                questions: [
                    "What are the four databases used in FinSolve's data layer and what is each used for?",
                    "What branch strategy does FinSolve follow in Git and what is each branch used for?",
                    "What is the minimum unit test coverage required before a pull request can be merged?",
                    "How are production releases deployed and how frequently?",
                    "What are the four severity levels for security vulnerabilities and their remediation timelines?",
                    "What deployment strategies are used for production releases?",
                    "What are the four bug severity classifications and their SLAs?",
                    "What are the three short-term AI/ML initiatives planned for Q2-Q4 2025?",
                    "How does FinSolve handle log retention — what are the hot, warm, and cold storage durations?",
                    "What compliance frameworks does FinSolve adhere to?"
                ]
            }
        ]
    },

    marketing: {
        title: "Marketing",
        sections: [
            {
                title: "Direct Lookups",
                questions: [
                    "What was the total marketing budget for 2024?",
                    "What was the customer acquisition cost (CAC) per new customer in 2024?",
                    "What was the Return on Ad Spend (ROAS) for digital campaigns in 2024?",
                    "What was the marketing spend in Q1 2024?",
                    "What was the customer acquisition target for Q2 2024?",
                    "How many new customers were actually acquired in Q3 2024?",
                    "What was the conversion rate achieved in Q4 2024?",
                    "How many customers were enrolled in the Q4 loyalty program?",
                    "What was the revenue target for Q3 2024?",
                    "How many enterprise accounts were targeted in Q4 ABM campaigns?"
                ]
            },
            {
                title: "Comparison & Analysis",
                questions: [
                    "Which quarter had the highest revenue target in 2024?",
                    "How did the CPA change across Q1, Q2, Q3, and Q4?",
                    "Which quarter first expanded into Southeast Asia and what were the key channels used?",
                    "In which quarter did FinSolve expand into Latin America and what merchant results were achieved?",
                    "What was the biggest reason customer acquisition fell short in Q1?",
                    "How did the ROI target compare to actual ROI across all four quarters?",
                    "What percentage of the 2024 marketing budget was allocated to digital marketing?",
                    "Which campaign in 2024 generated the highest ROI and by how much?",
                    "What were the top two recommendations made for Q1 2025?",
                    "How did customer retention rate targets progress from Q1 to Q4 2024?"
                ]
            }
        ]
    }
};