use std::cell::RefCell;
use std::sync::mpsc::{Receiver, Sender};
use std::time::Duration;

type ClaimPause = (Sender<()>, Receiver<()>);

thread_local! {
    static NEXT_CLAIM: RefCell<Option<ClaimPause>> = const { RefCell::new(None) };
}

pub fn pause_next_claim(entered: Sender<()>, resume: Receiver<()>) {
    NEXT_CLAIM.with(|pause| {
        assert!(pause.borrow_mut().replace((entered, resume)).is_none());
    });
}

pub(crate) fn pause_local_claim() {
    if let Some((entered, resume)) = NEXT_CLAIM.with(|pause| pause.borrow_mut().take()) {
        entered
            .send(())
            .expect("claim controller receives snapshot");
        resume
            .recv_timeout(Duration::from_secs(5))
            .expect("claim controller resumes CAS");
    }
}
