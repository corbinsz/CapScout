export default function Privacy() {
  return (
    <article className="legal-page">
      <p className="eyebrow">Placeholder</p>
      <h1>Privacy</h1>
      <p>
        This page describes what the app actually stores today. It is a placeholder for the owner
        to finish with a lawyer before a public launch. It is not a complete privacy policy.
      </p>
      <h2>Account data</h2>
      <p>
        If you create an account we store your email address, a hash of your password, the time
        you signed up, and whether your plan is free or Pro. We do not store the password itself.
        The session is an HttpOnly cookie. We do not put your email in the address bar or in
        application logs on purpose.
      </p>
      <h2>What we do not collect</h2>
      <p>
        Browsing scores does not require an account. We do not sell personal information. We do
        not run third-party advertising scripts. Map tiles and, when a key is configured, Street
        View photos are requested from their providers and those providers have their own policies.
      </p>
      <h2>Payments</h2>
      <p>
        Card numbers are not handled by CapScout. If the owner connects Stripe, Stripe hosts the
        checkout page and receives the payment details. We would store a Stripe customer id next
        to the account so a paid plan can be recognized.
      </p>
      <h2>Listing data</h2>
      <p>
        For-sale homes come from RentCast when an API key is set. A local development fallback can
        read a saved file on the server. Short-term rental history used to train the models comes
        from Inside Airbnb under their Creative Commons Attribution 4.0 license. Guest names are
        not shown.
      </p>
      <h2>Retention</h2>
      <p>
        Account rows stay until the operator deletes the database file or a person asks for the
        account to be removed. There is no self-serve delete button yet. That is a gap the owner
        should close before launch.
      </p>
    </article>
  );
}
