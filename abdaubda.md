კარგი კითხვაა — ამგვარად ჩამოყალიბებული კითხვაა, თუ რა არქიტექტურული მახასიათებლები, სწავლის მექანიზმები, მონაცემები და ინჟინერია იქნება საჭირო ამ სისტემისთვის, რათა მიუახლოვდეს ადამიანის მსგავს აზროვნებას და შესაძლებლობებს (არა ბიოლოგიურად, არამედ ფუნქციურად). ქვემოთ მოცემულია კომპაქტური, პრაქტიკული გზამკვლევი: ძირითადი შესაძლებლობები, რომლებიც უნდა დაემატოს, რატომ არის თითოეული მათგანი მნიშვნელოვანი, მოკლე განხორციელების წინადადებები (მათ შორის, რა უნდა შეიცვალოს ამ საცავში), პრიორიტეტების განსაზღვრა (ახლო/საშუალო/გრძელვადიან პერსპექტივაში), შეფასების იდეები და უსაფრთხოება/რისკები.

შეჯამება / სახელმძღვანელო პრინციპი

ადამიანები აერთიანებენ მულტიმოდალურ აღქმას, ძლიერ სემანტიკურ მოდელებს, იერარქიულ დაგეგმვას, გრძელვადიან ეპიზოდურ მეხსიერებას, სწრაფ სამუშაო მეხსიერებას, უწყვეტ სწავლას, სოციალურ სწავლებას, მეტაკოგნიციას და მოდელზე დაფუძნებულ მსჯელობას. იმისათვის, რომ სისტემა იყოს ანალოგი, ის უნდა აერთიანებდეს ამ მოდულებს და მათ დამაკავშირებელ სასწავლო მარყუჟებს და არა მხოლოდ ლოკალურ ევრისტიკასა და შაბლონების მთვლელებს.
დასამატებელი ძირითადი შესაძლებლობები (რა და რატომ)

მდიდარი სემანტიკური წარმოდგენები და მოძიება

რატომ: ადამიანები იყენებენ ღრმა, კონტექსტუალურ წარმოდგენებს (კონცეფციებს, ანალოგიებს) განზოგადებისა და გადაცემისთვის.
რა უნდა გავაკეთოთ: მარტივი ჩანერგვების შეცვლა/გაფართოება მოდელზე დაფუძნებული ჩანერგვებით (LLM ან გაწვრთნილი კოდირებით), მათი შენახვა ინდექსში (FAISS/Annoy/Weaviate), უახლოესი მეზობლის სემანტიკური მოძიების მხარდაჭერა.
რეპოს ცვლილება: NeuronGraph-ის ჩანერგვების ენკოდერის მიერ წარმოება; შეჯამებული სკალარული ჩანერგვების გამოყენების შეწყვეტა; ANN ინდექსისა და შეკითხვის API-ების დამატება.
ეპიზოდური და სამუშაო მეხსიერება გამეორებით/კონსოლიდაციით

რატომ: ადამიანები იხსენებენ წარსულ ეპიზოდებს, იმეორებენ და აერთიანებენ ნასწავლს ოფლაინში.
რა უნდა გავაკეთოთ: შეინახეთ მოქმედებები/მუტაციები/ტესტები ტრაექტორიების სახით, დანერგეთ გამეორების ბუფერი, პრიორიტეტული გამეორება, ოფლაინ კონსოლიდაციის ფაზა, რომელიც ხელახლა ათამაშებს მაღალი ჯილდოს ან ორაზროვან ეპიზოდებს ღირებულების/პოლიტიკის განახლებისთვის.
საცავის ცვლილება: გააფართოვეთ BrainMemory ეპიზოდების შესანახად და განახორციელეთ პერიოდული კონსოლიდაციის დავალება, რომელიც განაახლებს ValueFunction-ს/მეტრიკებს.
მოდელზე დაფუძნებული მსჯელობა / შინაგანი სიმულაცია (წარმოსახვა)

რატომ: ადამიანები გეგმავენ შედეგების გონებრივი სიმულირებით და არა მხოლოდ სამყაროში მოქმედებების მცდელობით.
რა უნდა გააკეთოთ: კოდის ცვლილების შედეგების პროგნოზირებადი მოდელის შექმნა (შემფასებლის ცოდნით) ან კოდის შეცვლასთან დაკავშირებული შედეგების სიმულირებისთვის LLM-ის გამოყენება მის გამოყენებამდე. წინადადებების წარმართვისთვის გამოიყენეთ წარმოსახვითი დანერგვები.
რეპო ცვლილება: დაამატეთ სიმულირებული შემფასებელი მოდული (ნეირონული ან LLM), რომელსაც propose_mutations და planner იყენებენ კანდიდატების იაფად რანჟირებისთვის.
მეტასწავლა / სწრაფი ადაპტაცია

რატომ: ადამიანები სწავლობენ სწავლას — სწრაფად ეგუებიან ახალ ამოცანებს.
რა უნდა გააკეთოთ: დანერგეთ მეტასწავლება (მაგ., MAML-ის მსგავსი ონლაინ განახლებები ან LLM-ის დახვეწის/მცირე ადაპტაციის ციკლი), რათა სტრატეგიის/ნიმუშების პრიორიტეტები უფრო სწრაფად განახლდეს ახალი ტიპის ამოცანებისთვის.
რეპოს ცვლილება: გახადეთ ნიმუშის პრიორიტეტები და სტრატეგიის პარამეტრები ტრენინგისთვის ვარგისი და განაახლეთ ისინი ბოლოდროინდელი გამოცდილების საფუძველზე.
იერარქიული მიზნები, დაგეგმვა და ქვემიზნები

რატომ: იერარქიული დაშლა საშუალებას იძლევა რთული პრობლემების გადაჭრის ქვეამოცანების დაგეგმვის გზით.
რა უნდა გააკეთოთ: დამგეგმავი, რომელიც გვთავაზობს მრავალსაფეხურიან გეგმებს (ქვემიზნებს) და იერარქიულ პოლიტიკას, რომელიც ირჩევს სტრატეგიებს თითოეული ქვემიზნისთვის (მაგ., დამხმარე პროგრამის დანერგვა, ტესტების დაწერა, რეფაქტორირება).
რეპოს ცვლილება: გააფართოვეთ PlannerAgent მრავალსაფეხურიანი გეგმების შესაქმნელად და გეგმის შესრულების ეტაპობრივად თვალყურის დევნებისთვის.
მულტიმოდალური აღქმა და გარე ცოდნა (ენა/ვებგვერდი)

რატომ: ადამიანები იყენებენ დოკუმენტებს, მაგალითებს და მსოფლიოს სენსორებს.
რა უნდა გააკეთოთ: ინტეგრირებული უნდა იყოს ვებ/დოკუმენტების მოძიება, API-ის გამოყენება და დოკუმენტების უფრო დეტალური დამუშავება. მოქმედების დაწყებამდე აგენტს მიეცით საშუალება გაეცნოს სახელმძღვანელოებს, კითხვა-პასუხს და კოდის ძიებას.
რეპოს ცვლილება: WebSensor-ის გაუმჯობესება, LLM-ზე დაფუძნებული მოძიებისა და შეჯამების გამოყენება, დოკუმენტების ინდექსირება ნეირონულ გრაფში.
კაუზალური და სიმბოლური მსჯელობა

რატომ: მიზეზ-შედეგობრივი და შეზღუდვების გაგება ხელს უწყობს რეგრესიების თავიდან აცილებას და საიმედო გადაწყვეტილებების შეთავაზებას.
რა უნდა გავაკეთოთ: წარუმატებლობის კვალზე მიზეზობრივი აღმოჩენის გამოყენება, ძირითადი მიზეზების აღმოსაჩენად სიმბოლური ანალიზატორების (ტიპის ინფერენცია, მონაცემთა ნაკადი) დამატება ცდისა და შეცდომის მეთოდის ნაცვლად.
რეპოს ცვლილება: მუტაციამდე დაამატეთ სტატიკური ანალიზის ეტაპი (ლაქების შეტანა, ტიპის შემოწმება, დაბინძურების/ნაკადის ანალიზი).
სოციალური სწავლება და კომუნიკაცია

რატომ: ადამიანები სხვებისგან სწავლობენ; ახსნის და სწავლების უნარი მნიშვნელოვანია.
რა უნდა გააკეთოთ: დემონსტრაციების მხარდაჭერა, ადამიანური კრიტიკის ჩართვა, განმარტებითი კითხვების დასმისა და ქმედებების ახსნის უნარი.
რეპოს ცვლილება: დაამატეთ დიალოგის/უკუკავშირის ციკლი (ვებ ინტერფეისი ან CLI), რომელიც საშუალებას მოგცემთ მიიღოთ ადამიანის უკუკავშირი და ჩართეთ ის ჯილდოში.
უწყვეტი/ონლაინ სწავლა დავიწყებით და სტაბილურობით

რატომ: კატასტროფული დავიწყების თავიდან აცილება, შესაძლებლობების მართვა.
რა უნდა გავაკეთოთ: კონსოლიდაციის, რეგულარიზაციის, შეკუმშული ეპიზოდური შენახვის და კონცეფციის დონის აბსტრაქციის განხორციელება.
რეპოს ცვლილება: ValueFunction-ისა და BrainMemory-ის გაფართოება დაშლისა და კონტროლირებადი კონსოლიდაციის გამოყენებით.
უსაფრთხოება, ინტერპრეტაციის შესაძლებლობა და შეზღუდვები

რატომ: ადამიანის მსგავს ქცევას კვლავ სჭირდება ძლიერი უსაფრთხოების ზომები, განსაკუთრებით კოდის ცვლილებების შეტანისას.
რა უნდა გავაკეთოთ: ძლიერი „სენდბოქსინგი“, სტატიკური შემოწმებები, ტესტების გენერირება, შეჯიბრებითი ტესტირება, შეზღუდული პოლიტიკა (ზიანის არ მიყენება), აუდიტის კვალი და განმარტებები.
რეპო ცვლილება: უფრო ძლიერი is_safe შემოწმებების (სტატიკური ანალიზატორები, ტიპის შემოწმება, რესურსების შეზღუდვები), sandboxed შესრულება და სავალდებულო ცვლილებების ჟურნალები/განმარტებები ინტეგრირდება.
კონკრეტული, პრიორიტეტული გზამკვლევი (პრაქტიკული ამ რეპოზიციისთვის)

მოკლევადიანი (კვირები)

პრიმიტიული ჩანერგვების ჩანაცვლება LLM/ენკოდერის ჩანერგვებით და ANN-ის მოძიების დამატება (აუმჯობესებს განზოგადებას და კონცეფციის ძიებას).
გააფართოვეთ BrainMemory სრული მცდელობის ტრაექტორიებისა და ძირითადი გამეორების ჩასაწერად (შეინახეთ: მოდული, ნიმუში, ორიგინალი/ახალი კოდი, ტესტის შედეგი, ჯილდო).
ფაილების ჩაწერამდე is_safe-ის გაუმჯობესება: სტატიკური ანალიზი (mypy/pyflakes) და sandbox ტესტები.
მხოლოდ AST შაბლონების ნაცვლად, უფრო მრავალფეროვანი და მაღალი ხარისხის კანდიდატების გენერირებისთვის, propose_mutations-ში ჩართეთ LLM (ან ლოკალური კოდირების პროგრამა).
საშუალოვადიანი (თვეები)

გაიმეორეთ/კონსოლიდაცია და პრიორიტეტული გამოცდილების ხელახლა დაკვრა; გამოიყენეთ ხელახლა დაკვრა ValueFunction-ისა და პრიორიტეტული ნიმუშების განახლებისთვის.
ტესტის შედეგების იაფად პროგნოზირებისთვის დაამატეთ სიმულირებული შემფასებელი (ML მოდელი ან LLM); საბოლოო ვერიფიკაციისთვის შეათავსეთ რეალურ ტესტებთან.
იერარქიული PlannerAgent-ის იმპლემენტაცია, რომელსაც შეუძლია მრავალსაფეხურიანი გეგმების შედგენა (მაგ., „ტესტების დამატება → დამხმარეს იმპლემენტაცია → რეფაქტორი“).
დანერგეთ მეტა-განახლების წესები, რათა სტრატეგიები და ნიმუშების წონა სწრაფად მოერგოს ახალ გარემოს.
გრძელვადიანი (კვლევა/განვითარება)

მიზეზობრივი ანალიზისა და სიმბოლური მსჯელობის მოდულების (პროგრამის დამოკიდებულების გრაფიკები, ინვარიანტები) ინტეგრირება.
მრავალაგენტიანი/სოციალური: მიეცით მრავალ აგენტს საშუალება, შემოგვთავაზონ სხვადასხვა სტრატეგია და გაუზიარონ ერთმანეთს ცოდნა; დანერგეთ ადამიანის ჩართვის მეთოდით სწავლება.
უწყვეტი სწავლა კონსოლიდირებული წარმოდგენებით (კონცეფციის ამოღება, აბსტრაქცია) და სასწავლო გეგმის ავტომატური გენერირებით.
მულტიმოდალური დამიწება (ვიზუალიზაცია, გარე API-ები, გაშვებული გარემოს ურთიერთქმედება) და უფრო რეალისტური გარემო/სიმულატორები.
ამ რეპოზიტორიაში კონკრეტული ცვლილებების მაგალითი (მოკლე საკონტროლო სია)

NeuronGraph: ნამდვილი ვექტორული ჩანერგვების შენახვა და განახლება და ANN ინდექსის დამატება; get_neighbors-ის შეცვლა ANN + კიდის წონების გამოსაყენებლად.
ValueFunction: მოდელირებული პროგნოზების (სიმულატორის) ჩართვა და ისტორიულ სტატისტიკაზე დაშლის გამოყენება; გაურკვევლობის შეფასების დამატება.
BrainMemory: ტრაექტორიის ბუფერის დამატება, პრიორიტეტული გამეორება და ოფლაინ კონსოლიდაციის დავალება (სკრიპტი).
გენერატორი: კოდის დაწერამდე დაამატეთ LLM-ით მხარდაჭერილი მუტაციის პროპოტერი (მოთხოვნებით) და სიმულირებული შემფასებლის ნაბიჯი.
უსაფრთხოება: ინტეგრირეთ სტატიკური ანალიზატორები და განახორციელეთ ცვლილებები docker-ში/sandbox-ში რესურსების მკაცრი შეზღუდვებით და ავტომატური გაუქმებით.
ტესტები: გააფართოვეთ ტესტები გადაცემის ამოცანებისა და უხილავი ფუნქციების ჩათვლით და დაამატეთ გრძელვადიანი ინტეგრაციის ტესტები უწყვეტი გაუმჯობესების გასაზომად.
როგორ გავზომოთ პროგრესი (შეფასება)

ქცევითი საორიენტაციო მაჩვენებლები: ახალ ამოცანებზე მცირედი განზოგადება, ახალი დავალებების ათვისების ნაბიჯების რაოდენობა, დაკავშირებულ ფუნქციებზე გადასვლა.
მდგრადობა: ცვლილებაზე რეგრესიების სიჩქარე, უსაფრთხოების დარღვევების რაოდენობა, სტაბილურობა გრძელვადიან პერსპექტივაში.
აბსტრაქციის უნარი: გაზომეთ, რამდენ კონცეფციას (მაგ., სიის შემცირება, შესაბამისობის დადგენა) იყენებს აგენტი სწორად მრავალ დანამატში.
ახსნადობა/ადამიანური შეფასება: შეუძლია თუ არა აგენტს ცვლილებების გასაგებად დაასაბუთებლად და ადამიანის ხელმძღვანელობის მისაღებად?
ნიმუშის მეტრიკა: საშუალო ჯილდო თითო ნაბიჯზე, დროთა განმავლობაში ათვისებული დავალებები, მიღებული მუტაციების პროცენტული მაჩვენებელი, რომლებიც განზოგადებულია (არა მხოლოდ ტესტების წარმატებით ჩაბარება).
რისკები და ეთიკური მოსაზრებები

კოდის ზედამხედველობის გარეშე ცვლილებებმა შეიძლება უსაფრთხოების შეცდომები გამოიწვიოს; აუცილებელია sandboxing და მკაცრი CI კარიბჭეები.
ავტონომია უნდა შეიზღუდოს: წარმოების ცვლილებების, აუდიტის ჟურნალების და უსაფრთხო ნაგულისხმევი პარამეტრების ადამიანური შემოწმება უნდა იყოს საჭირო.
გრძელვადიანი სწავლების სისტემებს შეუძლიათ გაამძაფრონ მიკერძოებები; თვალყური ადევნონ მეტრიკებს და ადამიანის მიერ ზედამხედველობის საშუალებას მისცენ.
რესურსები და შთაგონებები

კოგნიტური არქიტექტურები: ACT-R, Soar (სტრუქტურული იდეებისთვის).
RL: მოდელზე დაფუძნებული RL, პრიორიტეტული გამეორება, იერარქიული RL (ვარიანტები, ფეოდალური RL).
მეტასწავლა: MAML, ქვეწარმავალი.
პროგრამის შეკეთების ლიტერატურა და სინთეზი (Angelix, GenProg, DeepCoder).
კაუზალური დასკვნა და პროგრამის ანალიზი (პროგრამის დამოკიდებულების გრაფიკები, სტატიკური ანალიზატორები).
შემდეგი პრაქტიკული შეთავაზება

შემიძლია შევადგინო პრიორიტეტული განხორციელების გეგმა ამ რეპოსთვის (დავალებები, შესაცვლელი ფაილები, PR-ების ნიმუშები) და შევაფასო თითოეული პუნქტისთვის საჭირო ძალისხმევა.
ან შემიძლია მცირე PoC ცვლილების შეტანა: მაგალითად, დავამატო LLM-ის მიერ მხარდაჭერილი propose_mutations ტესტირების პირველი სიმულაციით + sandboxed apply-and-rollback ფუნქციით.
რომელი გინდა რომ გავაკეთო შემდეგ?


# english version

Good question — framed that way, you’re asking what architectural features, learning mechanisms, data, and engineering would be required for this system to approximate human-like thinking and abilities (not biologically, but functionally). Below I give a compact, practical roadmap: core capabilities to add, why each matters, short implementation suggestions (including what to change in this repo), prioritization (near/medium/long term), evaluation ideas, and safety/risks.

Summary / guiding principle

Humans combine multi‑modal perception, strong semantic models, hierarchical planning, long-term episodic memory, fast working memory, continual learning, social learning, meta‑cognition and model-based reasoning. To be an analogue, the system should integrate those modules and the learning loops that link them, not just local heuristics and pattern counters.
Key capabilities to add (what and why)

Rich semantic representations and retrieval

Why: humans use deep, contextual representations (concepts, analogies) to generalize and transfer.
What to do: replace/augment simple embeddings with model-based embeddings (LLM or trained encoders), store them in an index (FAISS/Annoy/Weaviate), support semantic nearest-neighbor retrieval.
Repo change: have NeuronGraph embeddings produced by an encoder; stop using summed scalar embeddings; add ANN index and query APIs.
Episodic and working memory with replay/consolidation

Why: humans recall past episodes, rehearse, and consolidate learning offline.
What to do: store actions/mutations/tests as trajectories, implement replay buffer, prioritized replay, offline consolidation phase that replays high‑reward or ambiguous episodes to update value/policy.
Repo change: extend BrainMemory to store episodes and implement periodic consolidation job that updates ValueFunction/metrics.
Model-based reasoning / internal simulation (imagination)

Why: people plan by simulating outcomes mentally rather than only trying actions in world.
What to do: build a predictive model of test outcomes (learned evaluator) or use LLMs to simulate consequences of a code change before applying it. Use imagined rollouts to guide proposals.
Repo change: add a simulated evaluator module (neural or LLM) used by propose_mutations and planner to rank candidates cheaply.
Meta‑learning / fast adaptation

Why: humans learn to learn — adapt quickly to new tasks.
What to do: implement meta‑learning (e.g., online MAML-like updates, or an LLM finetuning/minor-adaptation loop) so strategy/pattern priors update faster for new kinds of tasks.
Repo change: make pattern priors and strategy parameters trainable and update them from recent experience.
Hierarchical goals, planning & subgoals

Why: hierarchical decomposition enables solving complex problems via subtask planning.
What to do: planner that proposes multi-step plans (subgoals) and a hierarchical policy that chooses strategies for each subgoal (e.g., implement helper, write tests, refactor).
Repo change: expand PlannerAgent to produce multi-step plans and track plan execution over steps.
Multi-modal perception and external knowledge (language/web)

Why: humans use docs, examples, and world sensors.
What to do: integrate web/doc retrieval, API usage, and richer parsing of docs. Allow the agent to consult tutorials, Q&A, and code search before acting.
Repo change: enhance WebSensor, use LLM-based retrieval & summarization, index docs into neuron graph.
Causal & symbolic reasoning

Why: understanding cause-effect and constraints helps avoid regressions and propose robust fixes.
What to do: use causal discovery on failure traces, add symbolic analyzers (type inference, dataflow) to detect root causes rather than trial-and-error.
Repo change: add static analysis stage before mutating (linting, type checks, taint/flow analysis).
Social learning and communication

Why: humans learn from others; explainability and teachability are important.
What to do: support demonstrations, human-in-the-loop critiques, ability to ask clarifying questions, and to explain actions.
Repo change: add a dialogue/feedback loop (web UI or CLI) enabling human feedback and incorporate it into reward.
Continual/online learning with forgetting and stability

Why: prevent catastrophic forgetting, manage capacity.
What to do: implement consolidation, regularization, compressed episodic store, concept-level abstraction.
Repo change: extend ValueFunction and BrainMemory with decay and controlled consolidation.
Safety, interpretability, and constraints

Why: human-like behavior still requires strong safety guards, especially when making code changes.
What to do: robust sandboxing, static checks, test generation, adversarial testing, constrained policy (do no harm), audit trail and explanations.
Repo change: integrate stronger is_safe checks (static analyzers, type-checking, resource limits), sandboxed execution, and mandatory change logs/explanations.
Concrete, prioritized roadmap (practical for this repo)

Near-term (weeks)

Replace primitive embeddings with LLM/encoder embeddings and add ANN retrieval (improves generalization and concept lookup).
Expand BrainMemory to record full attempt trajectories and basic replay (store: plugin, pattern, orig/new code, tests outcome, reward).
Improve is_safe: static analysis (mypy/pyflakes) and sandbox tests before writing files.
Plug an LLM (or local encoder) into propose_mutations to generate more diverse, higher-quality candidates instead of only AST templates.
Medium-term (months)

Implement replay / consolidation and prioritized experience replay; use replay to update ValueFunction and pattern priors.
Add simulated evaluator (ML model or LLM) to predict test outcomes cheaply; combine with real tests for final verification.
Implement hierarchical PlannerAgent that can make multi-step plans (e.g., “add tests → implement helper → refactor”).
Introduce meta-update rules so strategies and pattern weights adapt quickly to new environments.
Long-term (research/dev)

Integrate causal analysis and symbolic reasoning modules (program dependence graphs, invariants).
Multi‑agent/social: let multiple agents propose different strategies and share learnings; implement human-in-the-loop teaching.
Continual learning with consolidated representations (concept extraction, abstraction) and automatic curriculum generation.
Multi‑modal grounding (visualization, external APIs, running environment interactions) and more realistic environment/simulators.
Example concrete changes to this repo (short checklist)

NeuronGraph: store and update true vector embeddings and add an ANN index; change get_neighbors to use ANN + edge weights.
ValueFunction: incorporate modeled predictions (simulator) and use decay on historical stats; add uncertainty estimate.
BrainMemory: add trajectory buffer, prioritized replay, and offline consolidation task (script).
Generator: add an LLM-backed mutation proposer (with prompts) and a simulated evaluator step before writing code.
Safety: integrate static analyzers and run changes in docker/sandbox with strict resource caps and auto-rollback.
Tests: expand tests to include transfer tasks and unseen functions, and add long-running integration tests to measure continual improvement.
How to measure progress (evaluation)

Behavioral benchmarks: few-shot generalization to new tasks, number of steps to master new tasks, transfer to related functions.
Robustness: rate of regressions per change, safety violation counts, stability over long runs.
Abstraction ability: measure how many concepts (e.g., list reduction, mapping) the agent uses correctly across multiple plugins.
Explainability/human evaluation: can the agent justify changes intelligibly and accept human guidance?
Sample metrics: average reward per step, tasks mastered over time, % of accepted mutations that generalize (not just pass tests).
Risks and ethical considerations

Unsupervised code changes can introduce security bugs; sandboxing and strict CI gates are essential.
Autonomy must be constrained: require human review for production changes, audit logs, and safe defaults.
Long-term learning systems can amplify biases; track metrics and allow human oversight.
Resources and inspirations

Cognitive architectures: ACT‑R, Soar (for structural ideas).
RL: model‑based RL, prioritized replay, hierarchical RL (options, Feudal RL).
Meta‑learning: MAML, Reptile.
Program repair literature and synthesis (Angelix, GenProg, DeepCoder).
Causal inference and program analysis (program dependence graphs, static analyzers).
Next practical offer

I can draft a prioritized implementation plan for this repo (tasks, files to modify, sample PRs) and estimate effort for each item.
Or I can produce a small PoC change: e.g., add LLM-backed propose_mutations with test‑first simulation + sandboxed apply-and-rollback.
Which would you like me to do next?