---
layout: page
permalink: /project/
title: Course Project
description: Guidelines and deliverables for course projects
---

Your class project offers an opportunity to apply deep learning / generative models to real-world problems. Projects will be completed in **teams of 2-4 students** and are worth **30% of your final grade**, consisting of the following deliverables:

1. **Proposal** (5%): 2 pages, excluding references.
2. **Midway Report** (5%): 4 pages, excluding references.
3. **Presentation** (5%): In-class presentation.
4. **Final Report** (15%): 8 pages, excluding references.

All submissions must adhere to the [ICML style template](https://www.overleaf.com/latex/templates/icml2025-template/dhxrkcgkvnkt).

---

## Team Formation

Teams of 3-4 students are strongly encouraged. Once your team is formed, email the instructor with the names of all members. If assistance is needed, contact the instructor to help find project partners.

---

## Deliverables and Deadlines

### **Proposal**

**Due:** Friday, October 17, 2025, at 11:59 PM via Canvas.

- **Content**:
  - Project title and team member list.
  - Problem statement and motivation (½ page).
  - Literature review of at least four relevant papers (~1 page).
  - Description of dataset(s) and planned activities.

- **Grading**:
  - 40%: Clear and concise description of the project.
  - 40%: Quality of literature survey.
  - 10%: Feasibility and detail of activity plan.
  - 10%: Writing quality.

---

### **Midway Report**

**Due:** Friday, November 7, 2025, at 11:59 PM via Canvas.

- **Purpose**:
  This report ensures you’ve made **real progress** on your project. Basically, your job in the midway report is to convince us that you will finish your project by the final due date. It's ok if your plans have changed from the proposal, but you should know that by now. It's also okay if your plans have failed or run into challenges, but you should have some indication of that by now.

  By now, you should have:
  - Implemented an initial version of your model or experiments.
  - Run at least some tests—even if they didn’t work.
  - Identified challenges and next steps.
    This is not just a plan—it must reflect **actual work done** so far.

- **Format & Length**:
  - **4 full pages**, excluding references. May **slightly spill onto a fifth page**, but be concise.
  - Use the **ICML template** again.
  - Put references must be on a separate page (**`\clearpage`** before).
  - **Figures & Tables**
    - Use **`figure`** for single-column figures, **`figure*`** for full-width figures. Figure sizes (including using **`\width=`** commands) will significantly change your report length, so size them properly.
    - Ensure figures are **numbered, referenced, and captioned** properly.
  - **Required Sections** (length of each sections will vary by project focus)
    - **Abstract (~150-200 words)**
      - Summary of the problem, approach, progress, key challenges, and next steps.
    - **Introduction (½ - 1 page)**
      - Problem statement and motivation.
      - Role of **deep learning** and/or **generative models** (PGMs).
    - **Related Work (~½ page)**
      - Summary of at least **four relevant papers** and how they relate to your work.
      - Don't just give a list of related papers. Instead, synthesize the related works into a narrative of where the field is and why your project is interesting.
    - **Methods (~1 page)**
      - Implemented work so far (not just plans).
      - Model structure, key assumptions, modifications.
      - Challenges encountered and adjustments made.
    - **Experiments (~1 page)**
      - Dataset(s) and experimental setup.
      - Preliminary results, even if unsuccessful.
    - **Conclusion & Updated Plan (~½ page)**
      - Summary of progress, challenges, and next steps.
      - Timeline for final report.

- **Grading**
  - **20%: Abstract, Introduction & Related Work** (Clarity, relevance, citations)
  - **30%: Methods & Implementation** (Actual work done, challenges, technical clarity)
  - **30%: Experiments & Preliminary Results** (Progress, results, adjustments)
  - **10%: Next Steps & Timeline** (Realistic plan, addressing issues)
  - **10%: Writing & Organization** (Clarity, structure, proper citations & figures)

---

### **In-Class Presentation**

**Date:** Monday, December 8 and Wednesday, December 10.

- **Format**: Each team will deliver a **4-minute presentation** in class, summarizing their project. Presentations will include a brief introduction, methods overview, key findings, and future directions. This is the only portion of the project that will have individual (per-person) grades, so make sure every group member contributes to the presentation.

- **Recommended Content** (but please adapt to fit your project): Title slide; Executive summary; Problem; State of the Field; Method; Results; Limitations / Next Steps; Takeaway

- **Presenting** We will use Prof. Lengerich's laptop to present slides in our classroom. **Submit your slides in presentation format (PPTX, Keynote, or PDF) 24 hours before your scheduled presentation.**

- **Grading**:
  - 30%: Clarity and engagement of presentation.
  - 40%: Depth and quality of content.
  - 30%: Use of visuals and adherence to time limits (3-5 minutes so we can fit all groups in the lecture time).

---

### **Final Report**

**Due:** Friday, December 12, 2025, at 11:59 PM via Canvas.

- **Content**:
  - ICML-style paper (8 pages).
  - Sections:
    - Abstract
    - Introduction
    - Background & Related Work (synthesize the related works into a narrative that places your work in context)
    - Methods
    - Experiments
    - Discussion (limitations, broader impact, etc.)
    - Conclusions
    - Reference (separate page using `\clearpage`)

- **Experimental expectations**:
  - At least one reasonable baseline.
  - At least one ablation / diagnostic (e.g. remove a term from your loss function, swap encoder, reduce data, etc).
  - Error / qualitative analysis (failure cases, calibration, representative samples, etc).
  - Report uncertainty where applicable (e.g., variance over training runs).

- **Grading**:
  - 10%: Problem definition and literature review.
  - 30%: Methodology (soundness and originality).
  - 30%: Experiments and results.
  - 10%: Analysis and discussion.
  - 20%: Writing quality.
