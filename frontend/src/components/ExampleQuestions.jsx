// Import the predefined example questions displayed on the Chat page.
import { exampleQuestions } from "../data/exampleQuestions";

// Displays example questions to help users understand
// what kinds of queries they can ask the AI assistant.
function ExampleQuestions() {

    return (

<section>

    {/* Page heading. */}
    <h1>Example Questions</h1>

    {/* Display every question category.  */}
    {
        Object.values(exampleQuestions).map((category) => (

            <section key={category.title}>

                {/* Display the category name. Example: HR, FINANCE, MARKETING */}
                <h2>{category.title}</h2>

                {/* Display every section within the current category. */}
                {
                    category.sections.map((section) => (

                        <section key={section.title}>

                            {/* Display the section name. Example: Basic Data Understanding, Comparison & Trend */}
                            <h3>{section.title}</h3>

                                {/* Display every example question within the current section. */}
                                <ol>
                                {
                                  section.questions.map((question, index) => (

                                      <p key={index}>
                                          {index + 1}. {question}
                                      </p>

                                    ))
                                }
                                </ol>

                        </section>

                    ))
                }

            </section>

        ))
    }

</section>

    );

}

export default ExampleQuestions;